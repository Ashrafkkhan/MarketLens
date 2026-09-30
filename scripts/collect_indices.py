#!/usr/bin/env python3
"""
collect_indices.py — MarketLens Phase 1 ETL CLI
================================================
Fetches historical daily OHLCV data for supported Indian broad-market
indices and upserts it into the `market_indices` table.

Supported indices:
  ^NSEI     → NIFTY 50
  ^NSEBANK  → NIFTY Bank
  ^CNXIT    → NIFTY IT

Usage::

    python scripts/collect_indices.py --start 2024-01-01 --end 2024-06-30
    python scripts/collect_indices.py --start 2024-01-01 --end 2024-06-30 --symbols ^NSEI ^NSEBANK
    python scripts/collect_indices.py --start 2024-01-01 --end 2024-06-30 --sleep 2.0

Options:
    --start    YYYY-MM-DD  Start date (inclusive). Required.
    --end      YYYY-MM-DD  End date   (inclusive). Required.
    --symbols  SYMBOL ...  Optional subset of index symbols to collect.
                           Defaults to all supported indices.
    --sleep    FLOAT       Seconds to sleep between API calls (default: 1.0).
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
from backend.services.data_providers.yfinance_provider import YFinanceProvider
from backend.services.ingestion_service import (
    SUPPORTED_INDICES,
    ingest_index_data,
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
logger = logging.getLogger("collect_indices")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect historical OHLCV data for Indian broad-market indices.",
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
        "--symbols",
        nargs="+",
        metavar="SYMBOL",
        default=None,
        help=(
            "Specific index symbols to collect "
            f"(choices: {', '.join(SUPPORTED_INDICES)}). "
            "Defaults to all supported indices."
        ),
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

    # Resolve index catalogue
    if args.symbols:
        unknown = set(args.symbols) - set(SUPPORTED_INDICES)
        if unknown:
            logger.error(
                "Unknown index symbol(s): %s. Supported: %s",
                ", ".join(sorted(unknown)),
                ", ".join(SUPPORTED_INDICES),
            )
            sys.exit(1)
        index_catalogue = {sym: SUPPORTED_INDICES[sym] for sym in args.symbols}
    else:
        index_catalogue = SUPPORTED_INDICES

    logger.info(
        "Starting index collection: %s → %s | indices=%s",
        args.start,
        args.end,
        list(index_catalogue.keys()),
    )

    provider = YFinanceProvider(sleep_seconds=args.sleep)

    total_fetched  = 0
    total_inserted = 0
    total_skipped  = 0
    total_errors   = 0

    with SessionLocal() as session:
        indices = list(index_catalogue.items())

        for i, (symbol, name) in enumerate(indices, start=1):
            logger.info(
                "Processing index %d/%d: %s (%s)",
                i, len(indices), symbol, name,
            )
            try:
                summary = ingest_index_data(
                    session=session,
                    provider=provider,
                    index_symbol=symbol,
                    index_name=name,
                    start=args.start,
                    end=args.end,
                )
                total_fetched  += summary["fetched"]
                total_inserted += summary["inserted"]
                total_skipped  += summary["skipped"]
                total_errors   += summary["errors"]
            except Exception as exc:
                logger.error(
                    "Unexpected error processing index %s: %s", symbol, exc
                )
                total_errors += 1

            if i < len(indices):
                time.sleep(args.sleep)

    logger.info(
        "Index collection complete — indices=%d, fetched=%d, inserted=%d, "
        "duplicates_skipped=%d, errors=%d.",
        len(index_catalogue),
        total_fetched,
        total_inserted,
        total_skipped,
        total_errors,
    )

    if total_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
