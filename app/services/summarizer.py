import json
import time

import openai
from google import genai
from google.genai.types import GenerateContentConfig

from app.config import settings
from app.logging import get_logger
from app.transcript import Transcript


logger = get_logger()

# Maximum characters per chunk for summarization
MAX_CHUNK_SIZE = 30000

# Schema returned by the structured summarizer
STRUCTURED_SUMMARY_SCHEMA = """
{
  "abstract": "<3-5 sentence overview of the talk>",
  "key_topics": ["<topic 1>", "<topic 2>", ...],
  "speakers": ["<name or 'Speaker 0' if unknown>", ...],
  "key_moments": [
    {"timestamp": "<HH:MM:SS or empty>", "topic": "<what is being discussed>"},
    ...
  ],
  "bitcoin_terms": ["<technical term mentioned>", ...]
}
"""


class SummarizerService:
    def __init__(self, provider="openai", model="gpt-4o", structured: bool = True):
        self.provider = provider
        self.model = model
        self.structured = structured  # Return JSON-structured summary
        if self.provider == "openai":
            self.client = openai
            self.client.api_key = settings.OPENAI_API_KEY
        elif self.provider == "google":
            self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
            if self.model == "gpt-4o":  # Default overwrite for google
                self.model = "gemini-2.0-flash"
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _split_into_chunks(
        self, text: str, max_size: int = MAX_CHUNK_SIZE
    ) -> list[str]:
        """Split text into chunks at paragraph boundaries."""
        if len(text) <= max_size:
            return [text]

        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = (
                    current_chunk + "\n\n" + para if current_chunk else para
                )

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process(self, transcript: Transcript, **kwargs):
        logger.info(
            f"Summarizing transcript with {self.provider} (model: {self.model})..."
        )
        text_to_summarize = transcript.outputs.get(
            "corrected_text", transcript.outputs["raw"]
        )

        text_length = len(text_to_summarize)
        logger.info(f"Text length for summarization: {text_length} characters")

        chunks = self._split_into_chunks(text_to_summarize)
        num_chunks = len(chunks)

        if num_chunks > 1:
            logger.info(
                f"Splitting text into {num_chunks} chunks for summarization..."
            )
            chunk_summaries = []

            for i, chunk in enumerate(chunks, 1):
                logger.info(
                    f"Summarizing chunk {i}/{num_chunks} ({len(chunk)} chars)..."
                )
                # For multi-chunk transcripts, extract plain-text partial summaries
                # then do a final structured pass over those.
                summary = self._summarize_text(chunk, is_chunk=True)
                if summary:
                    chunk_summaries.append(summary)
                    logger.info(
                        f"Chunk {i}/{num_chunks} summarization complete."
                    )

            if len(chunk_summaries) > 1:
                logger.info("Combining chunk summaries into final summary...")
                combined_text = "\n\n---\n\n".join(chunk_summaries)
                final_summary = self._summarize_text(
                    combined_text,
                    is_final=True,
                    title=transcript.source.title,
                    structured=self.structured,
                )
                transcript.summary = final_summary
            else:
                transcript.summary = (
                    chunk_summaries[0] if chunk_summaries else ""
                )
        else:
            summary = self._summarize_text(
                text_to_summarize,
                title=transcript.source.title,
                structured=self.structured,
            )
            transcript.summary = summary

            # Also store structured data separately so callers can parse it
            if self.structured:
                transcript.outputs["structured_summary"] = self._parse_structured(summary)

        logger.info(
            f"Summarization complete. Summary length: {len(transcript.summary)} chars"
        )

    def _summarize_text(
        self,
        text: str,
        is_chunk: bool = False,
        is_final: bool = False,
        title: str = None,
        structured: bool = False,
    ) -> str:
        """Summarize a piece of text, optionally returning a structured JSON summary."""
        if is_final:
            prompt = self._build_final_prompt(text, title, structured=structured)
        elif is_chunk:
            # Chunks always produce plain-text; structured pass happens at final merge
            prompt = self._build_chunk_prompt(text)
        else:
            prompt = self._build_full_prompt(text, title, structured=structured)

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=300,
                )
                return response.choices[0].message.content
            elif self.provider == "google":
                return self._call_with_retry(prompt, max_tokens=4096)
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return ""

    def _build_full_prompt(self, text: str, title: str = None, structured: bool = False) -> str:
        """Prompt for a single-chunk transcript (most common case)."""
        title_line = f'titled "{title}"' if title else ""

        if structured:
            return (
                f"You are a Bitcoin conference transcript analyst.\n\n"
                f"Analyse the following transcript {title_line} and return a JSON object "
                f"matching this schema:\n{STRUCTURED_SUMMARY_SCHEMA}\n\n"
                f"Guidelines:\n"
                f"- abstract: 3-5 clear sentences covering the main argument and conclusion.\n"
                f"- key_topics: 5-10 specific topics discussed (not generic like 'introduction').\n"
                f"- speakers: Real names if mentioned; otherwise use diarization labels.\n"
                f"- key_moments: Up to 10 significant moments with timestamps if available in the text.\n"
                f"- bitcoin_terms: Technical terms specific to Bitcoin/Lightning used in this talk.\n"
                f"Return ONLY the JSON — no markdown fences, no explanation.\n\n"
                f"--- Transcript ---\n\n{text}\n\n--- End ---"
            )
        else:
            return (
                f"Please summarize the following Bitcoin conference transcript {title_line}.\n\n"
                f"Provide a comprehensive summary covering:\n"
                f"- Main topics and arguments\n"
                f"- Key technical concepts discussed\n"
                f"- Important insights and conclusions\n\n"
                f"--- Transcript ---\n\n{text}\n\n--- End ---"
            )

    def _build_chunk_prompt(self, text: str) -> str:
        return (
            "Summarize the key points in this Bitcoin conference transcript section.\n"
            "Focus on: main topics, technical concepts, arguments, and important details.\n\n"
            f"{text}"
        )

    def _build_final_prompt(self, text: str, title: str = None, structured: bool = False) -> str:
        """Prompt to combine chunk summaries into one final summary."""
        title_line = f'titled "{title}"' if title else ""

        if structured:
            return (
                f"The following are section summaries from a Bitcoin conference transcript {title_line}.\n"
                f"Combine them into a single JSON object matching this schema:\n"
                f"{STRUCTURED_SUMMARY_SCHEMA}\n\n"
                f"Merge and deduplicate across sections. "
                f"Return ONLY the JSON — no markdown fences, no explanation.\n\n"
                f"--- Section summaries ---\n\n{text}\n\n--- End ---"
            )
        else:
            return (
                f"The following are summaries of different parts of a transcript {title_line}.\n"
                f"Combine them into a single coherent summary:\n\n"
                f"{text}\n\n"
                f"Provide a well-structured summary with the main topics and key insights."
            )

    @staticmethod
    def _parse_structured(summary_text: str) -> dict:
        """Try to parse the LLM output as JSON; fall back to raw string."""
        import re

        text = summary_text.strip()
        # Strip markdown code fences if present
        if "```" in text:
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()

        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {"raw_summary": summary_text}

    def _call_with_retry(self, prompt, max_tokens=4096, max_retries=4):
        """Call Gemini with exponential backoff on 503/429 errors."""
        config = GenerateContentConfig(max_output_tokens=max_tokens)
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                return response.text
            except Exception as e:
                err = str(e)
                if "limit: 0" in err or "expired" in err.lower() or "INVALID_ARGUMENT" in err:
                    raise  # Hard quota failure — no point retrying
                if ("503" in err or "429" in err) and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5  # 5, 10, 20, 40 seconds
                    logger.warning(f"Gemini rate limited (attempt {attempt+1}), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
