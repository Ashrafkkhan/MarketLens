#!/usr/bin/env python3
"""
collect_market_data.py — MarketLens Phase 1 ETL CLI
====================================================
Fetches historical daily OHLCV data for all active companies in the
database and upserts it into the `market_data` table.

Usage::

    python scripts/collect_market_data.py --start 2024-01-01 --end 2024-06-30
    python scripts/collect_market_data.py --start 2024-01-01 --end 2024-06-30 --tickers INFY RELIANCE TCS
    python scripts/collect_market_data.py --start 2024-01-01 --end 2024-06-30 --sleep 2.0

Options:
    --start    YYYY-MM-DD  Start date (inclusive). Required.
    --end      YYYY-MM-DD  End date   (inclusive). Required.
    --tickers  TICKER ...  Optional subset of tickers to process.
                           Defaults to all active companies in the DB.
    --sleep    FLOAT       Seconds to sleep between API calls (default: 1.0).
    --no-body              (placeholder; unused for market data, for CLI parity)
"""
import argparse
import logging
import sys
import time
from datetime import date

# Ensure the project root is on the path when run directly
import io
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output on Windows consoles (handles ₹, ™, etc. in news headlines)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from database.connection import SessionLocal
from backend.services.data_providers.yfinance_provider import YFinanceProvider
from backend.services.ingestion_service import (
    get_active_companies,
    ingest_market_data,
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
logger = logging.getLogger("collect_market_data")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect historical OHLCV market data for active companies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        required=True,
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Inclusive start date for data fetch.",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Inclusive end date for data fetch.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help="Specific tickers to process. Defaults to all active companies.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Seconds to sleep between yfinance API calls.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    if args.start > args.end:
        logger.error("--start must be before or equal to --end.")
        sys.exit(1)

    logger.info(
        "Starting market data collection: %s → %s", args.start, args.end
    )

    provider = YFinanceProvider(sleep_seconds=args.sleep)

    with SessionLocal() as session:
        # Resolve ticker list
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers]
            logger.info("Processing %d ticker(s) from CLI argument.", len(tickers))
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

        # Per-ticker ingestion
        total_fetched  = 0
        total_inserted = 0
        total_skipped  = 0
        total_errors   = 0

        for i, ticker in enumerate(tickers, start=1):
            logger.info(
                "Processing ticker %d/%d: %s", i, len(tickers), ticker
            )
            try:
                summary = ingest_market_data(
                    session=session,
                    provider=provider,
                    ticker=ticker,
                    start=args.start,
                    end=args.end,
                )
                total_fetched  += summary["fetched"]
                total_inserted += summary["inserted"]
                total_skipped  += summary["skipped"]
                total_errors   += summary["errors"]
            except Exception as exc:
                logger.error(
                    "Unexpected error processing ticker %s: %s", ticker, exc
                )
                total_errors += 1

            # Polite delay between tickers (provider already sleeps per download)
            if i < len(tickers):
                time.sleep(args.sleep)

    logger.info(
        "Collection complete — tickers=%d, fetched=%d, inserted=%d, "
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
