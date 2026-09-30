#!/usr/bin/env python3
"""
collect_news.py — MarketLens Phase 1 ETL CLI
============================================
Fetches financial news articles for active companies from the Google
News RSS feed and upserts them into the `news` table.

The RSS query format used is:
    {TICKER} NSE when:{N}d

Usage::

    python scripts/collect_news.py --start 2024-06-01 --end 2024-06-30
    python scripts/collect_news.py --start 2024-06-01 --end 2024-06-30 --tickers INFY TCS
    python scripts/collect_news.py --start 2024-06-01 --end 2024-06-30 --no-body
    python scripts/collect_news.py --start 2024-06-01 --end 2024-06-30 --sleep 2.0

Options:
    --start    YYYY-MM-DD  Start date (inclusive). Required.
    --end      YYYY-MM-DD  End date   (inclusive). Required.
    --tickers  TICKER ...  Optional subset of tickers to collect news for.
                           Defaults to all active companies in the DB.
    --sleep    FLOAT       Seconds to sleep between RSS/article fetches (default: 1.5).
    --no-body              Skip article body extraction (headline + metadata only).
                           Faster but stores no full text.
"""
import argparse
import logging
import sys
import time
from datetime import date

import io
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from database.connection import SessionLocal
from backend.services.data_providers.rss_news_provider import RSSNewsProvider
from backend.services.ingestion_service import (
    get_active_companies,
    ingest_news,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("collect_news")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect financial news articles for active companies via Google News RSS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        required=True,
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Inclusive start date for news collection.",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Inclusive end date for news collection.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help="Specific tickers to collect news for. Defaults to all active companies.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        metavar="SECONDS",
        help="Seconds to sleep between RSS and article fetch requests.",
    )
    parser.add_argument(
        "--no-body",
        action="store_true",
        default=False,
        help="Skip article body extraction. Only headline and metadata are stored.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------
def _build_rss_query(ticker: str) -> str:
    """
    Build a Google News RSS search query for a given ticker.

    Strips the '.NS' suffix for a cleaner query (Google News doesn't
    know about yfinance suffixes) and appends 'NSE' for Indian market context.

    Examples:
        'INFY.NS'     → 'INFY NSE'
        'RELIANCE.NS' → 'RELIANCE NSE'
        'INFY'        → 'INFY NSE'
    """
    base = ticker.split(".")[0].upper()
    return f"{base} NSE"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    if args.start > args.end:
        logger.error("--start must be before or equal to --end.")
        sys.exit(1)

    logger.info(
        "Starting news collection: %s → %s | fetch_body=%s",
        args.start,
        args.end,
        not args.no_body,
    )

    provider = RSSNewsProvider(
        sleep_seconds=args.sleep,
        fetch_article_body=not args.no_body,
    )

    with SessionLocal() as session:
        # Resolve ticker list
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers]
            logger.info(
                "Processing %d ticker(s) from CLI argument.", len(tickers)
            )
        else:
            companies = get_active_companies(session)
            tickers = [c.ticker for c in companies]
            logger.info(
                "Found %d active company/ticker(s) in the database.", len(tickers)
            )

        if not tickers:
            logger.warning(
                "No tickers to process. Add companies to the `companies` table first."
            )
            sys.exit(0)

        total_fetched  = 0
        total_inserted = 0
        total_skipped  = 0
        total_errors   = 0

        for i, ticker in enumerate(tickers, start=1):
            query = _build_rss_query(ticker)
            logger.info(
                "Processing ticker %d/%d: %s (query=%r)",
                i, len(tickers), ticker, query,
            )
            try:
                summary = ingest_news(
                    session=session,
                    provider=provider,
                    query=query,
                    start=args.start,
                    end=args.end,
                )
                total_fetched  += summary["fetched"]
                total_inserted += summary["inserted"]
                total_skipped  += summary["skipped"]
                total_errors   += summary["errors"]
            except Exception as exc:
                logger.error(
                    "Unexpected error collecting news for %s: %s", ticker, exc
                )
                total_errors += 1

            # Polite delay between tickers
            if i < len(tickers):
                time.sleep(args.sleep)

    logger.info(
        "News collection complete — tickers=%d, fetched=%d, inserted=%d, "
        "duplicates_skipped=%d, errors=%d.",
        len(tickers),
        total_fetched,
        total_inserted,
        total_skipped,
        total_errors,
    )

    if total_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
