"""Multilingual translation service.

Translates Bitcoin conference transcripts into:
- Indian regional languages (Hindi, Tamil, Telugu, Kannada, Malayalam,
  Bengali, Gujarati, Marathi) via Sarvam AI — purpose-built for these.
- European / global languages (Spanish, French, German, Portuguese,
  Japanese, Chinese) via Google Gemini, with automatic OpenAI fallback.

Architecture:
- Transcripts are split into chunks (≤ 3000 chars) to stay within API limits.
- Each chunk is translated independently; results are joined.
- The final translation is stored in transcript.outputs["translation_<lang_code>"].
"""

import time
from typing import Optional

import openai
import requests
from google import genai
from google.genai.types import GenerateContentConfig

from app.config import settings
from app.logging import get_logger
from app.transcript import Transcript


logger = get_logger()

# Languages handled by Sarvam AI (ISO 639-1 → Sarvam language code)
SARVAM_LANGUAGES: dict[str, str] = {
    "hi": "hi-IN",   # Hindi
    "ta": "ta-IN",   # Tamil
    "te": "te-IN",   # Telugu
    "kn": "kn-IN",   # Kannada
    "ml": "ml-IN",   # Malayalam
    "bn": "bn-IN",   # Bengali
    "gu": "gu-IN",   # Gujarati
    "mr": "mr-IN",   # Marathi
    "pa": "pa-IN",   # Punjabi
    "or": "or-IN",   # Odia
}

# Languages handled by Gemini
GEMINI_LANGUAGES: dict[str, str] = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ja": "Japanese",
    "zh": "Chinese (Simplified)",
    "ar": "Arabic",
    "ru": "Russian",
    "ko": "Korean",
    "it": "Italian",
}

ALL_SUPPORTED_LANGUAGES = {**SARVAM_LANGUAGES, **GEMINI_LANGUAGES}

SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
MAX_CHUNK_SIZE = 3000  # Characters per translation chunk


class TranslationService:
    """Translate a Transcript into one or more target languages.

    Usage:
        service = TranslationService()
        service.process(transcript, target_languages=["hi", "es", "fr"])
        # Results in transcript.outputs["translation_hi"], etc.
    """

    def __init__(self, llm_provider: str = "google", llm_model: str = "gemini-2.0-flash"):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self._gemini_client: Optional[genai.Client] = None


    def process(self, transcript: Transcript, target_languages: list[str], **kwargs):
        """Translate the transcript into each requested language.

        Args:
            transcript: The Transcript object (corrected text preferred).
            target_languages: List of ISO 639-1 codes, e.g. ["hi", "es"].
        """
        source_text = transcript.outputs.get(
            "corrected_text", transcript.outputs.get("raw", "")
        )
        if not source_text:
            logger.warning("No source text found for translation.")
            return

        for lang_code in target_languages:
            lang_code = lang_code.strip().lower()
            if lang_code not in ALL_SUPPORTED_LANGUAGES:
                logger.warning(f"Unsupported language code: {lang_code} — skipping.")
                continue

            logger.info(f"Translating transcript to '{lang_code}'...")
            try:
                translated = self._translate(source_text, lang_code)
                key = f"translation_{lang_code}"
                transcript.outputs[key] = translated
                logger.info(
                    f"Translation to '{lang_code}' complete "
                    f"({len(translated)} chars)."
                )
            except Exception as e:
                logger.error(f"Translation to '{lang_code}' failed: {e}")

    def translate_text(self, text: str, target_language: str) -> str:
        """Translate plain text — usable outside the full Transcript pipeline."""
        lang_code = target_language.strip().lower()
        if lang_code not in ALL_SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{lang_code}'. "
                f"Supported: {sorted(ALL_SUPPORTED_LANGUAGES)}"
            )
        return self._translate(text, lang_code)

    def _translate(self, text: str, lang_code: str) -> str:
        chunks = self._split_into_chunks(text)
        translated_chunks = []

        for i, chunk in enumerate(chunks, 1):
            if len(chunks) > 1:
                logger.info(f"Translating chunk {i}/{len(chunks)}...")

            if lang_code in SARVAM_LANGUAGES:
                result = self._translate_sarvam(chunk, lang_code)
            else:
                result = self._translate_global(chunk, lang_code)

            translated_chunks.append(result)

            # Avoid rate limits between chunks
            if i < len(chunks):
                time.sleep(1)

        return "\n\n".join(translated_chunks)

    def _translate_global(self, text: str, lang_code: str) -> str:
        """Translate a global language: try Gemini first, fall back to OpenAI."""
        try:
            return self._translate_gemini(text, lang_code)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower() or "expired" in err.lower():
                logger.warning(
                    f"Gemini quota/key issue for '{lang_code}', falling back to OpenAI..."
                )
                return self._translate_openai(text, lang_code)
            raise

    def _translate_sarvam(self, text: str, lang_code: str) -> str:
        """Call Sarvam AI translation API."""
        sarvam_lang = SARVAM_LANGUAGES[lang_code]
        api_key = self._get_sarvam_key()

        payload = {
            "input": text,
            "source_language_code": "en-IN",  # Source is English
            "target_language_code": sarvam_lang,
            "speaker_gender": "Male",
            "mode": "formal",
            "model": "mayura:v1",
            "enable_preprocessing": True,
        }
        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": api_key,
        }

        for attempt in range(4):
            try:
                resp = requests.post(
                    SARVAM_TRANSLATE_URL,
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                translated = data.get("translated_text") or data.get("output", "")
                if not translated:
                    raise ValueError(f"Empty translation response: {data}")
                return translated
            except requests.HTTPError as e:
                if e.response.status_code in (429, 503) and attempt < 3:
                    wait = 2 ** attempt * 5
                    logger.warning(
                        f"Sarvam rate limited (attempt {attempt + 1}), "
                        f"waiting {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("Sarvam translation failed after retries.")

    @staticmethod
    def _get_sarvam_key() -> str:
        key = settings.config.get("sarvam_api_key") or __import__("os").getenv("SARVAM_API_KEY")
        if not key:
            raise EnvironmentError(
                "SARVAM_API_KEY is not set. Add it to your .env file to use "
                "Indian language translation."
            )
        return key

    def _translate_gemini(self, text: str, lang_code: str) -> str:
        """Translate via Gemini with a structured prompt."""
        language_name = GEMINI_LANGUAGES[lang_code]
        client = self._get_gemini_client()

        prompt = (
            f"You are a professional translator specialising in Bitcoin and "
            f"blockchain technology.\n\n"
            f"Translate the following English text into {language_name}.\n\n"
            f"Rules:\n"
            f"- Preserve all technical Bitcoin terms (UTXO, Lightning Network, "
            f"  Taproot, Schnorr, mempool, BIP, etc.) in their original English form "
            f"  or their widely recognised translation in {language_name}.\n"
            f"- Preserve speaker labels (e.g. 'Speaker 0:') unchanged.\n"
            f"- Do NOT summarise or shorten the text.\n"
            f"- Return ONLY the translated text — no explanations or notes.\n\n"
            f"--- Text to translate ---\n\n{text}\n\n--- End of text ---"
        )

        config = GenerateContentConfig(
            max_output_tokens=8192,
            temperature=0.3,
        )

        for attempt in range(4):
            try:
                response = client.models.generate_content(
                    model=self.llm_model,
                    contents=prompt,
                    config=config,
                )
                return response.text.strip()
            except Exception as e:
                err = str(e)
                # Hard quota (limit: 0) or expired key — no point retrying
                if "limit: 0" in err or "expired" in err.lower() or "INVALID_ARGUMENT" in err:
                    raise
                if ("503" in err or "429" in err) and attempt < 3:
                    wait = 2 ** attempt * 5
                    logger.warning(
                        f"Gemini rate limited (attempt {attempt + 1}), "
                        f"waiting {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("Gemini translation failed after retries.")

    def _translate_openai(self, text: str, lang_code: str) -> str:
        """Translate via OpenAI gpt-4o-mini as fallback for global languages."""
        language_name = GEMINI_LANGUAGES[lang_code]
        prompt = (
            f"You are a professional translator specialising in Bitcoin and "
            f"blockchain technology.\n\n"
            f"Translate the following English text into {language_name}.\n\n"
            f"Rules:\n"
            f"- Preserve all technical Bitcoin terms (UTXO, Lightning Network, "
            f"  Taproot, Schnorr, mempool, BIP, etc.) in their original English form.\n"
            f"- Preserve speaker labels (e.g. 'Speaker 0:') unchanged.\n"
            f"- Do NOT summarise or shorten the text.\n"
            f"- Return ONLY the translated text — no explanations or notes.\n\n"
            f"--- Text to translate ---\n\n{text}\n\n--- End of text ---"
        )
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            timeout=120,
        )
        return response.choices[0].message.content.strip()

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        return self._gemini_client


    @staticmethod
    def _split_into_chunks(text: str, max_size: int = MAX_CHUNK_SIZE) -> list[str]:
        """Split at paragraph boundaries to keep context intact."""
        if len(text) <= max_size:
            return [text]

        chunks = []
        paragraphs = text.split("\n\n")
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 > max_size:
                if current:
                    chunks.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para

        if current:
            chunks.append(current.strip())

        return chunks
