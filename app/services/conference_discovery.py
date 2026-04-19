"""Conference discovery service.

Scrapes events.coinpedia.org for upcoming Bitcoin conferences, filters
by relevance, discovers their YouTube channels/playlists, and registers
them in the channel database for automatic scanning.
"""

import re
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from app.config import settings
from app.logging import get_logger
from app.services.database_service import get_database_service


logger = get_logger()

COINPEDIA_EVENTS_URL = "https://events.coinpedia.org/"

# Known Bitcoin speakers / researchers — if a name appears in an event's
# description or speaker list, that's a strong relevance signal
BITCOIN_SPEAKERS = {
    "adam back", "pieter wuille", "greg maxwell", "andrew poelstra",
    "luke dashjr", "matt corallo", "peter todd", "eric lombrozo",
    "jameson lopp", "nick szabo", "hal finney", "wladimir van der laan",
    "john newbery", "marco falke", "fanquake", "aj towns",
    "christian decker", "rusty russell", "laolu osuntokun", "roasbeef",
    "olaoluwa osuntokun", "lisa neigut", "alex bosworth",
    "stacker news", "lyn alden", "saifedean ammous", "michael saylor",
    "jack mallers", "marty bent", "stephan livera", "peter mccormack",
    "natalie brunell", "american hodl", "dylan leclair", "preston pysh",
}

# Keywords that indicate Bitcoin-technical content worth archiving
BITCOIN_KEYWORDS = {
    "bitcoin", "btc", "lightning", "lightning network", "taproot", "schnorr",
    "segwit", "utxo", "mempool", "timelock", "multisig", "script", "bip",
    "nostr", "layer 2", "l2", "sidechain", "fedimint", "cashu", "ecash",
    "mining", "hashrate", "difficulty", "proof of work", "pow", "stratum",
    "braidpool", "ocean", "block template", "node", "full node", "core dev",
    "bitcoin core", "bitcoin development", "bitcoin protocol", "bitcoin conference",
    "bitcoin summit", "bitcoin meetup", "bitdevs", "advancing bitcoin",
    "tab conf", "bitcoin++ ", "bitcoin 2025", "btc++",
}

# Keywords that indicate non-Bitcoin / to-skip content
SKIP_KEYWORDS = {
    "ethereum", "defi", "nft", "solana", "cardano", "polkadot", "avalanche",
    "bnb", "shitcoin", "altcoin", "web3", "metaverse", "doge", "xrp",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; BitScribe/1.0; +https://genesis-kb.com)"
    )
}


class ConferenceDiscoveryService:
    """Discovers Bitcoin conferences from events.coinpedia.org and
    auto-registers their YouTube channels for ingestion."""

    def __init__(self):
        self._db = get_database_service()
        self._session = requests.Session()
        self._session.headers.update(HEADERS)


    def run(self, max_pages: int = 5) -> dict:
        """Full discovery pipeline: scrape → filter → register channels.

        Args:
            max_pages: Maximum number of event listing pages to scrape.

        Returns:
            Summary dict with counts and errors.
        """
        logger.info("Starting conference discovery pipeline...")

        events = self._scrape_events(max_pages=max_pages)
        logger.info(f"Scraped {len(events)} total events.")

        bitcoin_events = [e for e in events if self._is_bitcoin_relevant(e)]
        logger.info(f"{len(bitcoin_events)} events passed Bitcoin relevance filter.")

        registered = 0
        skipped = 0
        errors = []

        for event in bitcoin_events:
            try:
                channel_url = self._find_youtube_channel(event)
                if not channel_url:
                    logger.debug(f"No YouTube channel found for: {event['name']}")
                    skipped += 1
                    continue

                channel_id = self._extract_channel_id(channel_url)
                if not channel_id:
                    logger.debug(f"Could not extract channel ID from {channel_url}")
                    skipped += 1
                    continue

                if self._channel_already_registered(channel_id):
                    logger.debug(f"Channel already registered: {channel_id}")
                    skipped += 1
                    continue

                self._register_channel(event, channel_id, channel_url)
                registered += 1
                logger.info(f"Registered channel for: {event['name']} ({channel_id})")

                # Polite scraping — don't hammer YouTube
                time.sleep(1)

            except Exception as e:
                msg = f"Error processing '{event.get('name', '?')}': {e}"
                logger.error(msg)
                errors.append(msg)

        summary = {
            "total_events_scraped": len(events),
            "bitcoin_events_found": len(bitcoin_events),
            "channels_registered": registered,
            "channels_skipped": skipped,
            "errors": errors,
        }
        logger.info(
            f"Conference discovery complete: {registered} new channels registered, "
            f"{skipped} skipped."
        )
        return summary

    def _scrape_events(self, max_pages: int = 5) -> list[dict]:
        """Scrape event listings from events.coinpedia.org.

        Returns:
            List of event dicts with keys: name, url, description, date,
            location, website.
        """
        events = []
        page = 1

        while page <= max_pages:
            url = COINPEDIA_EVENTS_URL if page == 1 else f"{COINPEDIA_EVENTS_URL}?page={page}"
            try:
                resp = self._session.get(url, timeout=15)
                resp.raise_for_status()
            except Exception as e:
                logger.warning(f"Failed to fetch page {page}: {e}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            page_events = self._parse_event_cards(soup)

            if not page_events:
                break

            events.extend(page_events)

            # Check for a next-page link
            next_link = soup.find("a", {"rel": "next"}) or soup.find(
                "a", string=re.compile(r"next|›|»", re.I)
            )
            if not next_link:
                break

            page += 1
            time.sleep(1)

        return events

    def _parse_event_cards(self, soup: BeautifulSoup) -> list[dict]:
        """Parse individual event cards from a listings page."""
        events = []

        # coinpedia uses article / div cards — try multiple selectors
        cards = (
            soup.find_all("article", class_=re.compile(r"event|card", re.I))
            or soup.find_all("div", class_=re.compile(r"event[-_]?(card|item|listing)", re.I))
            or soup.find_all("li", class_=re.compile(r"event", re.I))
        )

        for card in cards:
            try:
                event = self._parse_single_card(card)
                if event:
                    events.append(event)
            except Exception as e:
                logger.debug(f"Could not parse card: {e}")

        return events

    def _parse_single_card(self, card) -> Optional[dict]:
        """Extract structured data from one event card."""
        # Name / title
        title_el = (
            card.find(["h1", "h2", "h3", "h4"], class_=re.compile(r"title|name|heading", re.I))
            or card.find(["h1", "h2", "h3", "h4"])
        )
        name = title_el.get_text(strip=True) if title_el else ""
        if not name:
            return None

        # Event detail URL
        link_el = card.find("a", href=True)
        event_url = ""
        if link_el:
            href = link_el["href"]
            event_url = href if href.startswith("http") else urljoin(COINPEDIA_EVENTS_URL, href)

        # Description
        desc_el = card.find(class_=re.compile(r"desc|summary|excerpt|content", re.I))
        description = desc_el.get_text(strip=True) if desc_el else ""

        # Date
        date_el = card.find(["time", "span"], class_=re.compile(r"date|time", re.I))
        date_str = ""
        if date_el:
            date_str = date_el.get("datetime") or date_el.get_text(strip=True)

        # Location
        loc_el = card.find(class_=re.compile(r"loc|location|place|city", re.I))
        location = loc_el.get_text(strip=True) if loc_el else ""

        # External website link (often the conference's own site)
        website = ""
        for a in card.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and "coinpedia" not in href:
                website = href
                break

        return {
            "name": name,
            "url": event_url,
            "description": description,
            "date": date_str,
            "location": location,
            "website": website,
        }

    def _fetch_event_details(self, event: dict) -> dict:
        """Fetch the full event detail page and extract extra metadata."""
        if not event.get("url"):
            return event

        try:
            resp = self._session.get(event["url"], timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Try to grab a longer description
            body = soup.find(class_=re.compile(r"content|body|description|detail", re.I))
            if body:
                event["description"] = (event.get("description") or "") + " " + body.get_text(" ", strip=True)[:500]

            # Look for links to the conference's website
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http") and "coinpedia" not in href and not event.get("website"):
                    event["website"] = href

        except Exception as e:
            logger.debug(f"Could not fetch event details for {event['url']}: {e}")

        return event

    def _is_bitcoin_relevant(self, event: dict) -> bool:
        """Return True if the event is likely a technical Bitcoin event.

        Uses a two-tier filter:
        1. Reject if dominated by non-Bitcoin keywords (Ethereum, NFT, etc.)
        2. Pass if Bitcoin keywords OR known Bitcoin speakers appear in the text.
        """
        text = " ".join([
            event.get("name", ""),
            event.get("description", ""),
            event.get("location", ""),
            event.get("speakers", ""),
        ]).lower()

        # Immediate reject if skip keywords present and bitcoin isn't dominant
        if any(kw in text for kw in SKIP_KEYWORDS):
            bitcoin_count = text.count("bitcoin") + text.count("btc")
            skip_count = sum(text.count(kw) for kw in SKIP_KEYWORDS)
            if skip_count >= bitcoin_count:
                return False

        # Pass on keyword match
        if any(kw in text for kw in BITCOIN_KEYWORDS):
            return True

        # Also pass if a known Bitcoin speaker/researcher is mentioned
        if any(speaker in text for speaker in BITCOIN_SPEAKERS):
            logger.debug(f"Event '{event.get('name')}' passed via speaker expertise filter.")
            return True

        return False

    def _find_youtube_channel(self, event: dict) -> Optional[str]:
        """Try to find a YouTube channel or playlist URL for the event.

        Strategy:
        1. Check the event's own website for a YouTube link.
        2. Search YouTube Data API for the conference name.
        """
        # First try the conference website
        if event.get("website"):
            channel_url = self._scrape_youtube_link(event["website"])
            if channel_url:
                return channel_url

        # Fallback: YouTube Data API search
        if settings.config.get("youtube_api_key") or hasattr(settings, "_YOUTUBE_API_KEY"):
            try:
                channel_url = self._search_youtube(event["name"])
                if channel_url:
                    return channel_url
            except Exception:
                pass

        return None

    def _scrape_youtube_link(self, website_url: str) -> Optional[str]:
        """Look for a YouTube channel/playlist link on the given website."""
        try:
            resp = self._session.get(website_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "youtube.com" in href or "youtu.be" in href:
                    normalized = self._normalize_youtube_url(href)
                    if normalized:
                        return normalized

        except Exception as e:
            logger.debug(f"Could not scrape {website_url}: {e}")

        return None

    def _search_youtube(self, query: str) -> Optional[str]:
        """Use YouTube Data API to find a channel for the conference."""
        try:
            from googleapiclient.discovery import build

            youtube = build("youtube", "v3", developerKey=settings.YOUTUBE_API_KEY)
            response = (
                youtube.search()
                .list(
                    q=f"{query} bitcoin conference",
                    type="channel",
                    part="id,snippet",
                    maxResults=1,
                )
                .execute()
            )
            items = response.get("items", [])
            if items:
                channel_id = items[0]["id"]["channelId"]
                return f"https://www.youtube.com/channel/{channel_id}"
        except Exception as e:
            logger.debug(f"YouTube search failed for '{query}': {e}")

        return None

    @staticmethod
    def _normalize_youtube_url(url: str) -> Optional[str]:
        """Normalize a YouTube URL to a canonical channel/playlist form."""
        if not url:
            return None
        # Already a channel or playlist URL
        if re.search(r"youtube\.com/(channel|c|@|playlist)", url):
            return url.split("?")[0]  # strip query params
        # Short video link — not useful for channel scanning
        if "youtu.be" in url:
            return None
        # Watch URL — not a channel
        if "watch?v=" in url:
            return None
        return None

    @staticmethod
    def _extract_channel_id(channel_url: str) -> Optional[str]:
        """Extract the raw channel ID or handle from a YouTube URL."""
        patterns = [
            r"youtube\.com/channel/([^/?&]+)",
            r"youtube\.com/@([^/?&]+)",
            r"youtube\.com/c/([^/?&]+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, channel_url)
            if m:
                return m.group(1)
        return None

    def _channel_already_registered(self, channel_id: str) -> bool:
        """Check if a channel is already in the DB."""
        if not self._db.is_available:
            return False
        try:
            channels = self._db.list_channels() or []
            return any(c.get("channel_id") == channel_id for c in channels)
        except Exception:
            return False

    def _register_channel(self, event: dict, channel_id: str, channel_url: str):
        """Add the discovered channel to the monitored channels table."""
        if not self._db.is_available:
            logger.warning("Database not available — cannot register channel.")
            return

        channel_data = {
            "channel_id": channel_id,
            "channel_name": event["name"],
            "channel_url": channel_url,
            "description": event.get("description", "")[:500],
            "category": "conference",
            "priority": 2,  # Higher priority than generic channels
            "is_active": True,
        }
        self._db.add_channel(channel_data)
