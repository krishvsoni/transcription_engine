"""APScheduler background job scheduler.

Registers and runs recurring tasks:
  - Channel scan      every 6 hours   (scan YouTube channels for new videos)
  - Classify pending  every 6 hours   (classify newly discovered videos)
  - RSS poll          every 24 hours  (poll Bitcoin podcast RSS feeds)
  - Conference scrape every 12 hours  (scrape events.coinpedia.org)

Started on FastAPI application startup; shut down cleanly on shutdown.

Usage (in server.py):
    from app.scheduler import start_scheduler, stop_scheduler
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.logging import get_logger


logger = get_logger()

_scheduler: BackgroundScheduler | None = None


# Job functions — each catches its own exceptions so a single failure
# doesn't silence all future runs.

def _job_scan_channels():
    """Scan all active YouTube channels for new videos."""
    try:
        from app.services.channel_scanner import ChannelScanner
        result = ChannelScanner().scan_all_channels()
        logger.info(
            f"[scheduler] Channel scan complete: "
            f"{result.get('videos_discovered', 0)} videos discovered."
        )
    except Exception as e:
        logger.error(f"[scheduler] Channel scan failed: {e}")


def _job_classify_pending():
    """Classify all pending videos using LLM."""
    try:
        from app.services.content_classifier import ContentClassifier
        result = ContentClassifier().classify_all_pending()
        logger.info(
            f"[scheduler] Classification complete: "
            f"{result.get('videos_approved', 0)} approved, "
            f"{result.get('videos_rejected', 0)} rejected."
        )
    except Exception as e:
        logger.error(f"[scheduler] Classification failed: {e}")


def _job_poll_rss():
    """Poll all active Bitcoin podcast RSS feeds for new episodes."""
    try:
        from app.services.rss_poller import RSSPoller
        result = RSSPoller().poll_all()
        logger.info(
            f"[scheduler] RSS poll complete: "
            f"{result.get('new_episodes_found', 0)} new, "
            f"{result.get('episodes_queued', 0)} queued."
        )
    except Exception as e:
        logger.error(f"[scheduler] RSS poll failed: {e}")


def _job_discover_conferences():
    """Scrape events.coinpedia.org and register new Bitcoin conference channels."""
    try:
        from app.services.conference_discovery import ConferenceDiscoveryService
        result = ConferenceDiscoveryService().run(max_pages=3)
        logger.info(
            f"[scheduler] Conference discovery complete: "
            f"{result.get('channels_registered', 0)} new channels registered."
        )
    except Exception as e:
        logger.error(f"[scheduler] Conference discovery failed: {e}")


# Scheduler lifecycle

def start_scheduler():
    """Create and start the background scheduler.

    Job intervals (all configurable via config.ini):
      channel_scan_interval_hours      default 6
      classify_interval_hours          default 6
      rss_poll_interval_hours          default 24
      conference_scan_interval_hours   default 12
    """
    global _scheduler

    if _scheduler and _scheduler.running:
        logger.warning("[scheduler] Already running — skipping start.")
        return

    from app.config import settings

    def _hours(key: str, default: int) -> int:
        try:
            return int(settings.config.get(key, str(default)))
        except (ValueError, TypeError):
            return default

    scan_h = _hours("channel_scan_interval_hours", 6)
    classify_h = _hours("classify_interval_hours", 6)
    rss_h = _hours("rss_poll_interval_hours", 24)
    conf_h = _hours("conference_scan_interval_hours", 12)

    _scheduler = BackgroundScheduler(
        job_defaults={"coalesce": True, "max_instances": 1},
        timezone="UTC",
    )

    _scheduler.add_job(
        _job_scan_channels,
        trigger=IntervalTrigger(hours=scan_h),
        id="channel_scan",
        name="YouTube channel scan",
        replace_existing=True,
    )
    _scheduler.add_job(
        _job_classify_pending,
        trigger=IntervalTrigger(hours=classify_h),
        id="classify_pending",
        name="Classify pending videos",
        replace_existing=True,
    )
    _scheduler.add_job(
        _job_poll_rss,
        trigger=IntervalTrigger(hours=rss_h),
        id="rss_poll",
        name="RSS feed poll",
        replace_existing=True,
    )
    _scheduler.add_job(
        _job_discover_conferences,
        trigger=IntervalTrigger(hours=conf_h),
        id="conference_discovery",
        name="Conference discovery",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        f"[scheduler] Started. Jobs: "
        f"channel_scan every {scan_h}h, "
        f"classify every {classify_h}h, "
        f"rss_poll every {rss_h}h, "
        f"conference_discovery every {conf_h}h."
    )


def stop_scheduler():
    """Stop the scheduler gracefully on application shutdown."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("[scheduler] Stopped.")


def get_scheduler_status() -> dict:
    """Return the current scheduler state and next run times for each job."""
    if not _scheduler or not _scheduler.running:
        return {"running": False, "jobs": []}

    jobs = []
    for job in _scheduler.get_jobs():
        next_run = job.next_run_time
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run_at": next_run.isoformat() if next_run else None,
        })

    return {"running": True, "jobs": jobs}
