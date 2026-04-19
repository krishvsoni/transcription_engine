"""Conference discovery routes.

Exposes endpoints to trigger conference scraping from events.coinpedia.org
and auto-register discovered YouTube channels for ingestion.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.logging import get_logger


logger = get_logger()
router = APIRouter(tags=["Conference Discovery"])


class DiscoveryOptions(BaseModel):
    max_pages: int = 5


@router.post("/discover")
async def discover_conferences(options: DiscoveryOptions = DiscoveryOptions()):
    """Scrape events.coinpedia.org, filter for Bitcoin conferences, and
    register their YouTube channels for automatic scanning.

    Returns a summary with counts of events scraped, channels found, and
    channels registered.
    """
    from app.services.conference_discovery import ConferenceDiscoveryService

    try:
        service = ConferenceDiscoveryService()
        result = service.run(max_pages=options.max_pages)
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"Conference discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-languages")
async def list_supported_languages():
    """Return the full list of supported translation languages."""
    from app.services.translation_service import ALL_SUPPORTED_LANGUAGES, SARVAM_LANGUAGES, GEMINI_LANGUAGES

    return {
        "sarvam_languages": {
            code: {"sarvam_code": sarvam_code, "provider": "sarvam"}
            for code, sarvam_code in SARVAM_LANGUAGES.items()
        },
        "llm_languages": {
            code: {"language_name": name, "provider": "gemini"}
            for code, name in GEMINI_LANGUAGES.items()
        },
        "total": len(ALL_SUPPORTED_LANGUAGES),
    }
