"""RSS feed poller for Bitcoin podcasts.

Monitors a configurable list of Bitcoin podcast RSS feeds, detects new
episodes since the last poll, filters them by relevance, and submits them
to the transcription pipeline automatically.

State is persisted in a JSON sidecar file (rss_state.json) stored next to
the transcription metadata — no DB migration required.

Default feeds (can be extended via the API or config):
  - Stephan Livera Podcast
  - What Bitcoin Did
  - Bitcoin Audible
  - Bitcoin Optech Podcast
  - The Bitcoin Layer
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import feedparser
import requests

from app.config import settings
from app.logging import get_logger


logger = get_logger()

# Default Bitcoin podcast RSS feeds
DEFAULT_FEEDS: list[dict] = [
    {
        "id": "stephan_livera",
        "name": "Stephan Livera Podcast",
        "url": "https://stephanlivera.com/feed/podcast/",
        "loc": "podcast",
        "category": ["bitcoin", "economics"],
        "tags": ["bitcoin", "lightning", "austrian-economics"],
        "active": True,
    },
    {
        "id": "what_bitcoin_did",
        "name": "What Bitcoin Did",
        "url": "https://www.whatbitcoindid.com/feed/podcast",
        "loc": "podcast",
        "category": ["bitcoin"],
        "tags": ["bitcoin", "interviews"],
        "active": True,
    },
    {
        "id": "bitcoin_audible",
        "name": "Bitcoin Audible",
        "url": "https://feeds.transistor.fm/bitcoin-audible",
        "loc": "podcast",
        "category": ["bitcoin", "education"],
        "tags": ["bitcoin", "education"],
        "active": True,
    },
    {
        "id": "bitcoin_optech",
        "name": "Bitcoin Optech Podcast",
        "url": "https://bitcoinops.org/feed.xml",
        "loc": "podcast",
        "category": ["bitcoin", "development"],
        "tags": ["bitcoin", "development", "protocol"],
        "active": True,
    },
    {
        "id": "bitcoin_layer",
        "name": "The Bitcoin Layer",
        "url": "https://feeds.buzzsprout.com/1965419.rss",
        "loc": "podcast",
        "category": ["bitcoin", "macroeconomics"],
        "tags": ["bitcoin", "macroeconomics"],
        "active": True,
    },
    {
        "id": "tales_from_crypt",
        "name": "Tales from the Crypt",
        "url": "https://feeds.megaphone.fm/talesfromthecrypt",
        "loc": "podcast",
        "category": ["bitcoin"],
        "tags": ["bitcoin"],
        "active": True,
    },
]

STATE_FILENAME = "rss_state.json"

# Keywords that mark an episode as technically worth transcribing
TECHNICAL_KEYWORDS = {
    "lightning", "taproot", "schnorr", "utxo", "mempool", "script",
    "bip", "protocol", "layer 2", "multisig", "privacy", "coinjoin",
    "channel", "routing", "liquidity", "mining", "hashrate", "consensus",
    "bitcoin core", "development", "developer", "engineer", "technical",
    "op_", "musig", "miniscript", "psbt", "descriptor", "fedimint",
    "cashu", "ecash", "nostr", "ordinals", "inscriptions",
}

# Max duration in seconds for auto-transcription (4 hours)
MAX_EPISODE_DURATION_SECONDS = 4 * 3600


class RSSPoller:
    """Poll Bitcoin podcast RSS feeds and queue new episodes for transcription.

    State management:
      - Feed registry and last-poll timestamps stored in rss_state.json.
      - Each known episode GUID is tracked to prevent duplicate submissions.
      - New episodes are submitted to the local transcription API.
    """

    def __init__(self):
        self._state_path = self._resolve_state_path()
        self._state = self._load_state()
        self._server_url = (
            settings.TRANSCRIPTION_SERVER_URL or "http://localhost:8000"
        )

    # Public API

    def poll_all(self) -> dict:
        """Poll every active feed and queue new episodes.

        Returns:
            Summary with counts of episodes found, queued, skipped, errors.
        """
        logger.info("Starting RSS poll cycle...")
        feeds = self._get_active_feeds()

        total_new = 0
        total_queued = 0
        total_skipped = 0
        errors = []

        for feed in feeds:
            try:
                result = self._poll_feed(feed)
                total_new += result["new_episodes"]
                total_queued += result["queued"]
                total_skipped += result["skipped"]
            except Exception as e:
                msg = f"Error polling '{feed['name']}': {e}"
                logger.error(msg)
                errors.append(msg)

        self._save_state()

        summary = {
            "feeds_polled": len(feeds),
            "new_episodes_found": total_new,
            "episodes_queued": total_queued,
            "episodes_skipped": total_skipped,
            "errors": errors,
            "polled_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            f"RSS poll complete: {total_new} new, {total_queued} queued, "
            f"{total_skipped} skipped across {len(feeds)} feeds."
        )
        return summary

    def poll_feed_by_id(self, feed_id: str) -> dict:
        """Poll a single feed by its ID."""
        feed = next(
            (f for f in self._get_all_feeds() if f["id"] == feed_id), None
        )
        if not feed:
            raise ValueError(f"Feed not found: {feed_id}")
        result = self._poll_feed(feed)
        self._save_state()
        return result

    def add_feed(self, feed: dict) -> dict:
        """Add a new feed to the registry."""
        feed_id = feed.get("id") or self._slugify(feed["name"])
        feed["id"] = feed_id
        feed.setdefault("active", True)
        feed.setdefault("loc", "podcast")
        feed.setdefault("category", ["bitcoin"])
        feed.setdefault("tags", ["bitcoin"])

        feeds = self._state.get("feeds", [])
        if any(f["id"] == feed_id for f in feeds):
            raise ValueError(f"Feed with id '{feed_id}' already exists.")

        feeds.append(feed)
        self._state["feeds"] = feeds
        self._save_state()
        logger.info(f"Added RSS feed: {feed['name']} ({feed['url']})")
        return feed

    def remove_feed(self, feed_id: str) -> bool:
        """Remove a feed from the registry."""
        feeds = self._state.get("feeds", [])
        original_count = len(feeds)
        self._state["feeds"] = [f for f in feeds if f["id"] != feed_id]
        if len(self._state["feeds"]) == original_count:
            return False
        self._save_state()
        logger.info(f"Removed RSS feed: {feed_id}")
        return True

    def list_feeds(self) -> list[dict]:
        """Return all registered feeds with their last-poll timestamp."""
        feeds = self._get_all_feeds()
        result = []
        for feed in feeds:
            feed_copy = dict(feed)
            feed_state = self._state.get("feed_state", {}).get(feed["id"], {})
            feed_copy["last_polled_at"] = feed_state.get("last_polled_at")
            feed_copy["known_episodes"] = len(
                feed_state.get("known_guids", [])
            )
            result.append(feed_copy)
        return result

    # Core polling logic

    def _poll_feed(self, feed: dict) -> dict:
        """Poll one feed and queue any new episodes."""
        feed_id = feed["id"]
        logger.info(f"Polling feed: {feed['name']} ({feed['url']})")

        parsed = feedparser.parse(feed["url"])
        if parsed.bozo and not parsed.entries:
            raise RuntimeError(
                f"feedparser error for {feed['url']}: {parsed.bozo_exception}"
            )

        feed_state = self._state.setdefault("feed_state", {}).setdefault(
            feed_id, {"known_guids": [], "last_polled_at": None}
        )
        known_guids: set[str] = set(feed_state.get("known_guids", []))

        new_episodes = []
        for entry in parsed.entries:
            guid = entry.get("id") or entry.get("link") or entry.get("title", "")
            if guid in known_guids:
                continue

            audio_url = self._extract_audio_url(entry)
            if not audio_url:
                known_guids.add(guid)
                continue

            episode = self._parse_entry(entry, feed, audio_url)
            new_episodes.append((guid, episode))

        queued = 0
        skipped = 0

        for guid, episode in new_episodes:
            known_guids.add(guid)
            if not self._is_relevant(episode):
                logger.debug(
                    f"  Skipping (not relevant): {episode['title'][:60]}"
                )
                skipped += 1
                continue

            try:
                self._submit_episode(episode)
                queued += 1
                logger.info(f"  Queued: {episode['title'][:60]}")
                time.sleep(0.5)  # Polite rate limit
            except Exception as e:
                logger.error(
                    f"  Failed to queue '{episode['title'][:40]}': {e}"
                )
                skipped += 1

        feed_state["known_guids"] = list(known_guids)
        feed_state["last_polled_at"] = datetime.now(timezone.utc).isoformat()

        return {
            "feed_id": feed_id,
            "new_episodes": len(new_episodes),
            "queued": queued,
            "skipped": skipped,
        }

    # Episode parsing

    @staticmethod
    def _extract_audio_url(entry) -> Optional[str]:
        """Find the audio enclosure URL from an RSS entry."""
        AUDIO_TYPES = {
            "audio/mpeg", "audio/mp3", "audio/wav",
            "audio/x-m4a", "audio/mp4", "audio/ogg",
        }
        for link in entry.get("links", []):
            if link.get("rel") == "enclosure" and link.get("type") in AUDIO_TYPES:
                return link.get("href")
        # Fallback: enclosures list
        for enc in entry.get("enclosures", []):
            if enc.get("type", "") in AUDIO_TYPES:
                return enc.get("href") or enc.get("url")
        return None

    @staticmethod
    def _parse_entry(entry, feed: dict, audio_url: str) -> dict:
        """Extract structured metadata from a feed entry."""
        # Date
        pub = entry.get("published_parsed") or entry.get("updated_parsed")
        date_str = ""
        if pub:
            try:
                date_str = datetime(*pub[:3]).strftime("%Y-%m-%d")
            except Exception:
                pass

        # Duration (itunes:duration is usually "HH:MM:SS" or seconds)
        duration_sec = 0
        raw_duration = entry.get("itunes_duration", "")
        if raw_duration:
            parts = str(raw_duration).split(":")
            try:
                if len(parts) == 3:
                    duration_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                elif len(parts) == 2:
                    duration_sec = int(parts[0]) * 60 + int(parts[1])
                else:
                    duration_sec = int(raw_duration)
            except (ValueError, IndexError):
                pass

        # Episode number
        episode_num = None
        if hasattr(entry, "itunes_episode"):
            try:
                episode_num = int(entry.itunes_episode)
            except (ValueError, TypeError):
                pass

        return {
            "title": entry.get("title", "Untitled"),
            "url": audio_url,
            "link": entry.get("link", ""),
            "description": entry.get("summary", "")[:500],
            "date": date_str,
            "duration_sec": duration_sec,
            "episode": episode_num,
            "loc": feed.get("loc", "podcast"),
            "category": feed.get("category", ["bitcoin"]),
            "tags": feed.get("tags", ["bitcoin"]),
            "feed_name": feed.get("name", ""),
        }

    # Relevance filtering

    def _is_relevant(self, episode: dict) -> bool:
        """Return True if episode is worth auto-transcribing."""
        # Duration guard — skip episodes that are too long
        dur = episode.get("duration_sec", 0)
        if dur > 0 and dur > MAX_EPISODE_DURATION_SECONDS:
            logger.debug(
                f"  Skipping (too long: {dur}s): {episode['title'][:50]}"
            )
            return False

        text = (
            (episode.get("title") or "") + " " +
            (episode.get("description") or "")
        ).lower()

        return any(kw in text for kw in TECHNICAL_KEYWORDS)

    
    # Pipeline submission

    def _submit_episode(self, episode: dict):
        """POST the episode to the local transcription pipeline."""
        data = {
            "source": episode["url"],
            "loc": episode["loc"],
            "title": episode["title"],
            "date": episode["date"],
            "tags": episode["tags"],
            "category": episode["category"],
            "deepgram": "true",
            "diarize": "true",
            "markdown": "true",
            "correct": "true",
            "summarize": "true",
            "llm_provider": settings.LLM_PROVIDER,
        }
        if episode.get("episode"):
            data["episode"] = str(episode["episode"])

        resp = requests.post(
            f"{self._server_url}/transcription/add_to_queue/",
            data=data,
            timeout=30,
        )
        resp.raise_for_status()

    # Feed registry helpers

    def _get_all_feeds(self) -> list[dict]:
        """Return all feeds — custom ones from state, fallback to defaults."""
        stored = self._state.get("feeds")
        if stored is not None:
            return stored
        # First run — seed with defaults
        self._state["feeds"] = list(DEFAULT_FEEDS)
        self._save_state()
        return self._state["feeds"]

    def _get_active_feeds(self) -> list[dict]:
        return [f for f in self._get_all_feeds() if f.get("active", True)]

    # State persistence

    def _resolve_state_path(self) -> Path:
        metadata_dir = settings.TSTBTC_METADATA_DIR or "metadata"
        path = Path(metadata_dir) / STATE_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load_state(self) -> dict:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Could not load RSS state: {e} — starting fresh.")
        return {}

    def _save_state(self):
        try:
            self._state_path.write_text(
                json.dumps(self._state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Could not save RSS state: {e}")

    @staticmethod
    def _slugify(name: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
