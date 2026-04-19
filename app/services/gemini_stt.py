"""Gemini STT provider.

Uses the Google Gemini 1.5 Flash model's native audio understanding to
produce transcripts with speaker diarization and word-level timestamps.
Follows the same interface as Deepgram and SmallestAI services.
"""

import os
import time
from pathlib import Path

from google import genai
from google.genai import types as genai_types

from app.config import settings
from app.data_writer import DataWriter
from app.logging import get_logger
from app.media_processor import MediaProcessor
from app.transcript import Transcript


logger = get_logger()

# Gemini's file API accepts up to 2 GB; audio is uploaded then referenced
MAX_AUDIO_BYTES = 500 * 1024 * 1024  # 500 MB safety limit


class GeminiSTT:
    """Transcribe audio using Gemini 1.5 Flash (audio understanding mode).

    The workflow:
    1. Upload the audio file to the Gemini Files API.
    2. Send a prompt asking for a verbatim transcript with speaker labels
       and timestamps.
    3. Parse the structured response into the same raw-text format used
       by Whisper / Deepgram outputs.
    """

    MODEL = "gemini-2.0-flash"

    def __init__(self, diarize: bool, upload: bool, data_writer: DataWriter):
        self.diarize = diarize
        self.upload = upload
        self.data_writer = data_writer
        self.language = settings.config.get("language", "en")
        self.one_sentence_per_line = settings.config.getboolean(
            "one_sentence_per_line", True
        )
        self.max_audio_length = 3600.0  # 60 minutes in seconds
        self.processor = MediaProcessor(chunk_length=1200.0)
        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    def audio_to_text(self, audio_file: str, chunk=None) -> dict:
        """Transcribe audio_file using Gemini and return a result dict.

        The returned dict mirrors Whisper's output format so downstream
        post-processing (correction, summarization) is unchanged:
            {
                "text": "<full transcript>",
                "segments": [{"start": float, "end": float, "text": str, "speaker": str}, ...]
            }
        """
        logger.info(
            f"Transcribing audio {f'(chunk {chunk}) ' if chunk else ''}"
            f"to text using Gemini STT [{self.language}]..."
        )

        file_size = os.path.getsize(audio_file)
        if file_size > MAX_AUDIO_BYTES:
            raise ValueError(
                f"Audio file too large for Gemini Files API: {file_size} bytes"
            )

        # Upload file to Gemini Files API
        gemini_file = self._upload_audio(audio_file)

        try:
            prompt = self._build_prompt()
            result = self._call_with_retry(gemini_file, prompt)
            return self._parse_response(result)
        finally:
            # Clean up the uploaded file to avoid storage quota leaks
            try:
                self._client.files.delete(name=gemini_file.name)
            except Exception:
                pass

    def _upload_audio(self, audio_file: str):
        """Upload audio to Gemini Files API and wait until it's ACTIVE."""
        mime_type = self._guess_mime_type(audio_file)
        logger.info(f"Uploading audio to Gemini Files API ({mime_type})...")

        with open(audio_file, "rb") as f:
            gemini_file = self._client.files.upload(
                file=f,
                config=genai_types.UploadFileConfig(
                    mime_type=mime_type,
                    display_name=Path(audio_file).name,
                ),
            )

        # Poll until ACTIVE (usually < 30 s for audio)
        for _ in range(30):
            gemini_file = self._client.files.get(name=gemini_file.name)
            state = str(getattr(gemini_file, "state", "")).upper()
            if state == "ACTIVE" or "ACTIVE" in state:
                break
            if "FAILED" in state:
                raise RuntimeError(
                    f"Gemini file upload failed: {gemini_file}"
                )
            time.sleep(3)
        else:
            raise TimeoutError("Gemini file did not become ACTIVE in time.")

        logger.info(f"Audio uploaded: {gemini_file.name}")
        return gemini_file

    def _build_prompt(self) -> str:
        diarize_instruction = (
            "Label each speaker as 'Speaker 0', 'Speaker 1', etc. "
            "Prefix every line with the speaker label when multiple speakers are present.\n"
            if self.diarize
            else ""
        )

        return (
            "You are a professional transcription service. "
            "Transcribe the audio file VERBATIM — do not summarize, paraphrase, or skip any words.\n\n"
            f"{diarize_instruction}"
            "Format each segment on its own line with a leading timestamp in [HH:MM:SS] format, "
            "for example:\n"
            "[00:00:05] Speaker 0: Hello everyone, welcome to the conference.\n"
            "[00:00:12] Speaker 1: Thank you for having me.\n\n"
            "Rules:\n"
            "- Include ALL spoken words, including filler words (um, uh, you know).\n"
            "- Do NOT add commentary or notes.\n"
            "- Do NOT translate — transcribe in the original language.\n"
            "- Use correct punctuation.\n"
            "- For Bitcoin technical terms (UTXO, Taproot, Lightning, Schnorr, etc.) "
            "  use correct capitalization.\n"
        )

    def _call_with_retry(self, gemini_file, prompt: str, max_retries: int = 4) -> str:
        """Call Gemini generate_content with exponential backoff."""
        config = genai_types.GenerateContentConfig(
            max_output_tokens=65536,
            temperature=0.0,  # Deterministic — we want verbatim transcript
        )
        contents = [
            genai_types.Part.from_uri(
                file_uri=gemini_file.uri,
                mime_type=gemini_file.mime_type,
            ),
            prompt,
        ]

        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.MODEL,
                    contents=contents,
                    config=config,
                )
                return response.text
            except Exception as e:
                if ("503" in str(e) or "429" in str(e)) and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5
                    logger.warning(
                        f"Gemini rate limited (attempt {attempt + 1}), waiting {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise

    def _parse_response(self, text: str) -> dict:
        """Parse Gemini's timestamped transcript into segments + full text.

        Expected line format:
            [HH:MM:SS] Speaker N: sentence text
        """
        import re

        segments = []
        full_lines = []

        # Match lines like: [00:01:23] Speaker 0: some text
        pattern = re.compile(
            r"^\[(\d{1,2}):(\d{2}):(\d{2})\]\s*"   # timestamp
            r"(?:(Speaker\s+\d+|[^:]{1,40}):\s*)?"  # optional speaker label
            r"(.+)$",                                 # text
            re.MULTILINE,
        )

        prev_start = 0.0
        for m in pattern.finditer(text):
            h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            start_sec = h * 3600 + mn * 60 + s
            speaker = (m.group(4) or "").strip()
            seg_text = m.group(5).strip()

            # Close the previous segment's end time
            if segments:
                segments[-1]["end"] = start_sec

            segment = {
                "start": float(start_sec),
                "end": float(start_sec + 10),  # Default; overwritten by next segment
                "text": seg_text,
                "speaker": speaker,
            }
            segments.append(segment)

            label = f"{speaker}: " if speaker else ""
            full_lines.append(f"{label}{seg_text}")

        # If Gemini returned plain text without timestamps, use it verbatim
        if not segments:
            logger.warning(
                "Gemini STT: no timestamped segments found in response; "
                "using raw text."
            )
            return {"text": text.strip(), "segments": []}

        full_text = "\n".join(full_lines)
        return {"text": full_text, "segments": segments}


    def write_to_json_file(self, result: dict, transcript: Transcript):
        """Write Gemini STT raw output to disk."""
        output_file = self.data_writer.write_json(
            data=result,
            file_path=transcript.output_path_with_title,
            filename="gemini_stt",
        )
        logger.info(f"(gemini_stt) Model output stored at: {output_file}")

        if transcript.metadata_file:
            self.data_writer.write_to_json_file_at_path(
                {"gemini_stt_output": output_file},
                transcript.metadata_file,
            )

    def process_transcript(self, result: dict, transcript: Transcript):
        """Convert Gemini output dict → transcript.outputs['raw'] text."""
        raw_text = result.get("text", "")

        # Format with diarization if segments carry speaker info
        if self.diarize and result.get("segments"):
            lines = []
            for seg in result["segments"]:
                speaker = seg.get("speaker", "")
                text = seg.get("text", "").strip()
                if speaker:
                    lines.append(f"{speaker}: {text}")
                else:
                    lines.append(text)
            raw_text = "\n".join(lines)

        transcript.outputs["raw"] = raw_text

    @staticmethod
    def _guess_mime_type(file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        mapping = {
            ".mp3": "audio/mpeg",
            ".mp4": "audio/mp4",
            ".m4a": "audio/mp4",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".webm": "audio/webm",
        }
        return mapping.get(ext, "audio/mpeg")
