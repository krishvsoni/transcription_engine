"""Sarvam AI STT provider.

Uses Sarvam's saarika:v2 model for speech-to-text, optimised for
Indian-accented English and Indian languages. Falls back gracefully
to plain text output when diarization is not supported.

Interface mirrors Deepgram and GeminiSTT so it slots into the existing
Transcription orchestrator with a sarvam_stt=True flag.
"""

import os
import time
from pathlib import Path

import requests

from app.config import settings
from app.data_writer import DataWriter
from app.logging import get_logger
from app.transcript import Transcript


logger = get_logger()

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"


class SarvamSTT:
    """Transcribe audio using Sarvam AI saarika:v2.

    Best suited for:
    - Indian-accented English
    - Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati,
      Marathi, Punjabi, Odia audio

    Output format mirrors Whisper / Deepgram:
        {
            "text": "<full transcript>",
            "segments": [{"start": float, "end": float, "text": str, "speaker": str}]
        }
    """

    MODEL = "saarika:v2"

    def __init__(self, diarize: bool, upload: bool, data_writer: DataWriter):
        self.diarize = diarize
        self.upload = upload
        self.data_writer = data_writer
        self.language = settings.config.get("language", "en")
        self._api_key = self._get_api_key()

    def audio_to_text(self, audio_file: str, chunk=None) -> dict:
        """Transcribe audio_file using Sarvam saarika:v2."""
        logger.info(
            f"Transcribing audio {f'(chunk {chunk}) ' if chunk else ''}"
            f"using Sarvam STT [{self.language}]..."
        )

        result = self._transcribe(audio_file)
        return result

    def _transcribe(self, audio_file: str) -> dict:
        """POST audio to Sarvam STT API and parse response."""
        mime = self._guess_mime(audio_file)

        # Sarvam STT expects multipart/form-data
        with open(audio_file, "rb") as f:
            files = {"file": (Path(audio_file).name, f, mime)}
            data = {
                "model": self.MODEL,
                "with_timestamps": "true",
                "with_diarization": "true" if self.diarize else "false",
            }
            headers = {"api-subscription-key": self._api_key}

            for attempt in range(4):
                try:
                    f.seek(0)
                    resp = requests.post(
                        SARVAM_STT_URL,
                        files=files,
                        data=data,
                        headers=headers,
                        timeout=300,
                    )
                    resp.raise_for_status()
                    break
                except requests.HTTPError as e:
                    if e.response.status_code in (429, 503) and attempt < 3:
                        wait = 2 ** attempt * 5
                        logger.warning(f"Sarvam STT rate limited, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise

        payload = resp.json()
        return self._parse_response(payload)

    def _parse_response(self, payload: dict) -> dict:
        """Parse Sarvam's STT response into standard format."""
        # Sarvam returns: { "transcript": "...", "timestamps": [...] }
        # or { "text": "..." } depending on model version
        transcript = (
            payload.get("transcript")
            or payload.get("text")
            or payload.get("output", "")
        )

        segments = []
        timestamps = payload.get("timestamps") or payload.get("segments") or []

        for ts in timestamps:
            start = float(ts.get("start", 0))
            end = float(ts.get("end", start + 5))
            text = ts.get("text") or ts.get("transcript") or ts.get("word", "")
            speaker = ts.get("speaker", "")
            if text:
                segments.append({
                    "start": start,
                    "end": end,
                    "text": text,
                    "speaker": speaker,
                })

        if not transcript and segments:
            transcript = " ".join(s["text"] for s in segments)

        return {"text": transcript, "segments": segments}

    def write_to_json_file(self, result: dict, transcript: Transcript):
        """Write Sarvam STT raw output to disk."""
        output_file = self.data_writer.write_json(
            data=result,
            file_path=transcript.output_path_with_title,
            filename="sarvam_stt",
        )
        logger.info(f"(sarvam_stt) Model output stored at: {output_file}")

        if transcript.metadata_file:
            self.data_writer.write_to_json_file_at_path(
                {"sarvam_stt_output": output_file},
                transcript.metadata_file,
            )

    def process_transcript(self, result: dict, transcript: Transcript):
        """Convert Sarvam output dict → transcript.outputs['raw'] text."""
        raw_text = result.get("text", "")

        if self.diarize and result.get("segments"):
            lines = []
            for seg in result["segments"]:
                speaker = seg.get("speaker", "")
                text = seg.get("text", "").strip()
                lines.append(f"{speaker}: {text}" if speaker else text)
            raw_text = "\n".join(lines)

        transcript.outputs["raw"] = raw_text

    @staticmethod
    def _guess_mime(file_path: str) -> str:
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

    @staticmethod
    def _get_api_key() -> str:
        key = settings.config.get("sarvam_api_key") or os.getenv("SARVAM_API_KEY")
        if not key:
            raise EnvironmentError(
                "SARVAM_API_KEY is not set. Add it to your .env file."
            )
        return key
