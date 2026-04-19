"""Translation routes.

Exposes endpoints to translate transcripts or arbitrary text into
supported languages using Sarvam AI (Indian languages) or Gemini (others).
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.logging import get_logger


logger = get_logger()
router = APIRouter(tags=["Translation"])


class TranslateTextRequest(BaseModel):
    text: str = Field(..., description="Text to translate (English source).")
    target_language: str = Field(
        ...,
        description="ISO 639-1 language code, e.g. 'hi', 'es', 'fr', 'ta'.",
    )


class TranslateTranscriptRequest(BaseModel):
    transcript_id: Optional[str] = Field(
        None, description="DB UUID of the transcript to translate."
    )
    raw_text: Optional[str] = Field(
        None, description="Supply raw text directly instead of a DB lookup."
    )
    target_languages: list[str] = Field(
        ...,
        description="List of ISO 639-1 codes, e.g. ['hi', 'es', 'fr'].",
    )


@router.get("/languages")
async def list_languages():
    """List all supported translation languages and their providers."""
    from app.services.translation_service import (
        ALL_SUPPORTED_LANGUAGES,
        GEMINI_LANGUAGES,
        SARVAM_LANGUAGES,
    )

    return {
        "supported_languages": {
            code: {
                "provider": "sarvam" if code in SARVAM_LANGUAGES else "gemini",
                "label": SARVAM_LANGUAGES.get(code) or GEMINI_LANGUAGES.get(code, ""),
            }
            for code in sorted(ALL_SUPPORTED_LANGUAGES)
        },
        "total": len(ALL_SUPPORTED_LANGUAGES),
    }


@router.post("/text")
async def translate_text(request: TranslateTextRequest):
    """Translate a short piece of English text into the target language.

    Useful for testing or translating titles/descriptions.
    """
    from app.services.translation_service import TranslationService

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")

    try:
        service = TranslationService()
        translated = service.translate_text(request.text, request.target_language)
        return {
            "status": "success",
            "source_language": "en",
            "target_language": request.target_language,
            "translated_text": translated,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcript")
async def translate_transcript(request: TranslateTranscriptRequest):
    """Translate a full transcript into one or more languages.

    Accepts either a transcript_id (DB lookup) or raw_text directly.
    Translations are returned in the response body as a dict keyed by
    language code, e.g. {"hi": "...", "es": "..."}.
    """
    from app.services.translation_service import ALL_SUPPORTED_LANGUAGES, TranslationService
    from app.transcript import Transcript

    if not request.transcript_id and not request.raw_text:
        raise HTTPException(
            status_code=400,
            detail="Provide either transcript_id or raw_text.",
        )

    # Validate language codes early
    unsupported = [
        lang for lang in request.target_languages
        if lang.strip().lower() not in ALL_SUPPORTED_LANGUAGES
    ]
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language codes: {unsupported}. "
                   f"GET /translation/languages for the full list.",
        )

    # Build a minimal Transcript-like object for the service
    if request.raw_text:
        source_text = request.raw_text
    else:
        # TODO: look up transcript by ID from DB when DB service is wired
        raise HTTPException(
            status_code=501,
            detail="DB-based transcript lookup not yet implemented. Use raw_text.",
        )

    try:
        service = TranslationService()
        translations = {}
        for lang_code in request.target_languages:
            lang_code = lang_code.strip().lower()
            translated = service.translate_text(source_text, lang_code)
            translations[lang_code] = translated

        return {
            "status": "success",
            "source_language": "en",
            "translations": translations,
        }
    except Exception as e:
        logger.error(f"Transcript translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
