#!/usr/bin/env python3
"""
extract_events.py — MarketLens Phase 2 ETL CLI (100% Local / Offline)
======================================================================
Reads un-processed news articles from the `news` table, runs each through
the local hybrid NLP pipeline (Qwen2.5-1.5B-Instruct + FinBERT), and writes
structured Event records into the `events` table.

"Un-processed" = news rows whose id does NOT appear in events.news_id.
Progress is committed every --commit-every rows so partial runs resume safely.

Usage::

    python scripts/extract_events.py --limit 20
    python scripts/extract_events.py --limit 100 --commit-every 10
    python scripts/extract_events.py --limit 50 --log-level DEBUG

Options:
    --limit         Max articles to process in this run (default: 50).
    --sleep         Seconds between articles — use 0 for local inference (default: 0.2).
    --commit-every  DB commit interval in successful inserts (default: 10).
    --log-level     DEBUG | INFO | WARNING | ERROR (default: INFO).
"""
import argparse
import io
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# UTF-8 stdout for Windows consoles (handles Rs, TM etc. in news headlines)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from database.connection import SessionLocal
from database.models import Event, News


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract structured financial events from raw news using "
            "local Qwen2.5-1.5B-Instruct (extraction) + FinBERT (sentiment)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of news articles to process in this run.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        metavar="SECONDS",
        help="Pause between articles. Use 0.0 for maximum local throughput.",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=10,
        metavar="N",
        help="Commit to DB after every N successful event inserts.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _fetch_unprocessed_news(session, limit: int) -> list[News]:
    """
    Return up to *limit* News rows that have no corresponding Event yet.
    Uses a NOT EXISTS correlated subquery for efficiency on large datasets.
    """
    subq = (
        session.query(Event.news_id)
        .filter(Event.news_id == News.id)
        .correlate(News)
        .exists()
    )
    return (
        session.query(News)
        .filter(~subq)
        .order_by(News.published_at.desc())
        .limit(limit)
        .all()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("extract_events")

    # --- Initialise service (models load lazily on first article) ------------
    logger.info(
        "Initialising EventExtractorService (Qwen2.5-1.5B-Instruct + FinBERT)..."
    )
    from backend.services.event_extractor import EventExtractorService
    service = EventExtractorService()

    # --- Fetch unprocessed news ----------------------------------------------
    with SessionLocal() as session:
        unprocessed = _fetch_unprocessed_news(session, args.limit)

    total = len(unprocessed)
    if total == 0:
        logger.info("No unprocessed news articles found. Nothing to do.")
        sys.exit(0)

    logger.info(
        "Found %d unprocessed article(s) to process (limit=%d).",
        total, args.limit,
    )

    # --- Processing loop ------------------------------------------------------
    succeeded     = 0
    failed        = 0
    pending_commit: list[Event] = []

    with SessionLocal() as session:
        for i, news_row in enumerate(unprocessed, start=1):
            logger.info(
                "[%d/%d] news_id=%-6d | %s",
                i, total, news_row.id,
                news_row.headline[:90],
            )

            # --- Extract (Qwen + FinBERT) ------------------------------------
            try:
                event_data = service.extract(
                    news_id=news_row.id,
                    headline=news_row.headline,
                    article_body=news_row.article,
                    session=session,
                )
            except Exception as exc:
                logger.error(
                    "[%d/%d] Extraction FAILED for news_id=%d: %s",
                    i, total, news_row.id, exc,
                )
                failed += 1
                time.sleep(args.sleep)
                continue

            # --- Persist Event -----------------------------------------------
            event = Event(**event_data)
            session.add(event)
            pending_commit.append(event)
            succeeded += 1

            logger.info(
                "         ticker=%-12s  type=%-20s  sentiment=%-8s  "
                "finbert=[pos=%.3f neg=%.3f neu=%.3f]",
                str(event_data.get("ticker") or "None"),
                event_data["event_type"],
                event_data["gemini_sentiment"],
                event_data["finbert_positive"],
                event_data["finbert_negative"],
                event_data["finbert_neutral"],
            )

            # --- Incremental commit every N rows -----------------------------
            if len(pending_commit) >= args.commit_every:
                try:
                    session.commit()
                    logger.info(
                        "[Commit] Saved %d events (running total: %d).",
                        len(pending_commit), succeeded,
                    )
                    pending_commit.clear()
                except Exception as exc:
                    logger.error(
                        "DB commit failed - rolling back %d rows: %s",
                        len(pending_commit), exc,
                    )
                    session.rollback()
                    succeeded -= len(pending_commit)
                    failed    += len(pending_commit)
                    pending_commit.clear()

            if i < total and args.sleep > 0:
                time.sleep(args.sleep)

        # --- Final flush -----------------------------------------------------
        if pending_commit:
            try:
                session.commit()
                logger.info(
                    "[Final Commit] Saved remaining %d events.", len(pending_commit)
                )
                pending_commit.clear()
            except Exception as exc:
                logger.error("Final DB commit failed: %s", exc)
                session.rollback()
                succeeded -= len(pending_commit)
                failed    += len(pending_commit)

    # --- Summary --------------------------------------------------------------
    logger.info(
        "Extraction complete - total=%d, succeeded=%d, failed=%d.",
        total, succeeded, failed,
    )
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
