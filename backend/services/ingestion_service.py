"""
MarketLens Ingestion Service.

Orchestrates data fetching (via provider classes) and idempotent
database upserts using PostgreSQL ON CONFLICT DO NOTHING semantics.

Handles:
  - MarketData  — conflict on (ticker, timestamp)
  - MarketIndex — conflict on (index_symbol, timestamp)
  - News        — conflict on (url)

All DB operations are batched per-ticker for performance and wrapped in
individual transactions so one bad batch doesn't roll back others.
"""
import logging
from datetime import date
from typing import Optional, Sequence

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from database.models import Company, MarketData, MarketIndex, News
from backend.services.data_providers.base import (
    BaseMarketProvider,
    BaseNewsProvider,
    IndexRecord,
    MarketRecord,
    NewsRecord,
)

logger = logging.getLogger(__name__)

# Batch size for bulk inserts — keeps individual statements manageable
_BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# Index catalogue
# ---------------------------------------------------------------------------

#: Supported Indian broad-market indices.
#: Keys are yfinance symbols; values are human-readable display names.
SUPPORTED_INDICES: dict[str, str] = {
    "^NSEI":    "NIFTY 50",
    "^NSEBANK": "NIFTY Bank",
    "^CNXIT":   "NIFTY IT",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunked(seq: list, size: int):
    """Yield successive *size*-length chunks from *seq*."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _upsert_market_data_batch(
    session: Session,
    records: list[MarketRecord],
) -> tuple[int, int]:
    """
    Bulk-insert a batch of MarketRecord rows using ON CONFLICT DO NOTHING.

    Returns:
        Tuple of (rows_inserted, rows_skipped).
    """
    if not records:
        return 0, 0

    rows = [
        {
            "ticker":    r.ticker,
            "timestamp": r.timestamp,
            "open":      r.open,
            "high":      r.high,
            "low":       r.low,
            "close":     r.close,
            "volume":    r.volume,
        }
        for r in records
    ]

    stmt = pg_insert(MarketData).values(rows)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=["ticker", "timestamp"]
    )

    result = session.execute(stmt)
    session.commit()

    inserted = result.rowcount if result.rowcount != -1 else len(rows)
    skipped  = len(rows) - inserted
    return inserted, skipped


def _upsert_index_batch(
    session: Session,
    records: list[IndexRecord],
) -> tuple[int, int]:
    """
    Bulk-insert a batch of IndexRecord rows using ON CONFLICT DO NOTHING.

    Returns:
        Tuple of (rows_inserted, rows_skipped).
    """
    if not records:
        return 0, 0

    rows = [
        {
            "index_name":   r.index_name,
            "index_symbol": r.index_symbol,
            "timestamp":    r.timestamp,
            "open":         r.open,
            "high":         r.high,
            "low":          r.low,
            "close":        r.close,
            "volume":       r.volume,
        }
        for r in records
    ]

    stmt = pg_insert(MarketIndex).values(rows)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=["index_symbol", "timestamp"]
    )

    result = session.execute(stmt)
    session.commit()

    inserted = result.rowcount if result.rowcount != -1 else len(rows)
    skipped  = len(rows) - inserted
    return inserted, skipped


def _upsert_news_batch(
    session: Session,
    records: list[NewsRecord],
) -> tuple[int, int]:
    """
    Bulk-insert a batch of NewsRecord rows using ON CONFLICT DO NOTHING.
    Conflict target is the unique URL constraint.

    Returns:
        Tuple of (rows_inserted, rows_skipped).
    """
    if not records:
        return 0, 0

    rows = [
        {
            "headline":     r.headline,
            "article":      r.article,
            "source":       r.source,
            "published_at": r.published_at,
            "url":          r.url,
        }
        for r in records
    ]

    stmt = pg_insert(News).values(rows)
    stmt = stmt.on_conflict_do_nothing(index_elements=["url"])

    result = session.execute(stmt)
    session.commit()

    inserted = result.rowcount if result.rowcount != -1 else len(rows)
    skipped  = len(rows) - inserted
    return inserted, skipped


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------

def ingest_market_data(
    session: Session,
    provider: BaseMarketProvider,
    ticker: str,
    start: date,
    end: date,
) -> dict:
    """
    Fetch and ingest daily OHLCV data for a single equity ticker.

    Args:
        session:   SQLAlchemy Session (caller is responsible for lifecycle).
        provider:  Concrete BaseMarketProvider implementation.
        ticker:    Ticker symbol (NSE suffix appended automatically).
        start:     Inclusive start date.
        end:       Inclusive end date.

    Returns:
        Summary dict with keys: ticker, fetched, inserted, skipped, errors.
    """
    summary = {
        "ticker":   ticker,
        "fetched":  0,
        "inserted": 0,
        "skipped":  0,
        "errors":   0,
    }

    try:
        records = provider.fetch_ohlcv(ticker, start, end)
    except Exception as exc:
        logger.error("[%s] Provider fetch_ohlcv failed: %s", ticker, exc)
        summary["errors"] += 1
        return summary

    summary["fetched"] = len(records)

    if not records:
        logger.info("[%s] No records to insert.", ticker)
        return summary

    total_inserted = 0
    total_skipped  = 0

    for batch in _chunked(records, _BATCH_SIZE):
        try:
            ins, skip = _upsert_market_data_batch(session, batch)
            total_inserted += ins
            total_skipped  += skip
        except Exception as exc:
            logger.error(
                "[%s] DB upsert failed for batch of %d records: %s",
                ticker,
                len(batch),
                exc,
            )
            session.rollback()
            summary["errors"] += 1

    summary["inserted"] = total_inserted
    summary["skipped"]  = total_skipped

    logger.info(
        "[%s] Ingestion complete — fetched=%d, inserted=%d, skipped(duplicate)=%d.",
        ticker,
        summary["fetched"],
        summary["inserted"],
        summary["skipped"],
    )
    return summary


def ingest_index_data(
    session: Session,
    provider: BaseMarketProvider,
    index_symbol: str,
    index_name: str,
    start: date,
    end: date,
) -> dict:
    """
    Fetch and ingest daily OHLCV data for a broad market index.

    Args:
        session:       SQLAlchemy Session.
        provider:      Concrete BaseMarketProvider implementation.
        index_symbol:  yfinance symbol (e.g. '^NSEI').
        index_name:    Human-readable name (e.g. 'NIFTY 50').
        start:         Inclusive start date.
        end:           Inclusive end date.

    Returns:
        Summary dict with keys: index_symbol, fetched, inserted, skipped, errors.
    """
    summary = {
        "index_symbol": index_symbol,
        "fetched":      0,
        "inserted":     0,
        "skipped":      0,
        "errors":       0,
    }

    try:
        records = provider.fetch_index_ohlcv(index_symbol, index_name, start, end)
    except Exception as exc:
        logger.error("[%s] Provider fetch_index_ohlcv failed: %s", index_symbol, exc)
        summary["errors"] += 1
        return summary

    summary["fetched"] = len(records)

    if not records:
        logger.info("[%s] No index records to insert.", index_symbol)
        return summary

    total_inserted = 0
    total_skipped  = 0

    for batch in _chunked(records, _BATCH_SIZE):
        try:
            ins, skip = _upsert_index_batch(session, batch)
            total_inserted += ins
            total_skipped  += skip
        except Exception as exc:
            logger.error(
                "[%s] DB upsert failed for index batch of %d records: %s",
                index_symbol,
                len(batch),
                exc,
            )
            session.rollback()
            summary["errors"] += 1

    summary["inserted"] = total_inserted
    summary["skipped"]  = total_skipped

    logger.info(
        "[%s] Index ingestion complete — fetched=%d, inserted=%d, skipped=%d.",
        index_symbol,
        summary["fetched"],
        summary["inserted"],
        summary["skipped"],
    )
    return summary


def ingest_news(
    session: Session,
    provider: BaseNewsProvider,
    query: str,
    start: date,
    end: date,
) -> dict:
    """
    Fetch and ingest news articles for a given search query.

    Args:
        session:   SQLAlchemy Session.
        provider:  Concrete BaseNewsProvider implementation.
        query:     Search string (e.g. 'INFY NSE').
        start:     Inclusive start date.
        end:       Inclusive end date.

    Returns:
        Summary dict with keys: query, fetched, inserted, skipped, errors.
    """
    summary = {
        "query":    query,
        "fetched":  0,
        "inserted": 0,
        "skipped":  0,
        "errors":   0,
    }

    try:
        records = provider.fetch_news(query, start, end)
    except Exception as exc:
        logger.error("Provider fetch_news failed for query=%r: %s", query, exc)
        summary["errors"] += 1
        return summary

    summary["fetched"] = len(records)

    if not records:
        logger.info("No news records to insert for query=%r.", query)
        return summary

    total_inserted = 0
    total_skipped  = 0

    for batch in _chunked(records, _BATCH_SIZE):
        try:
            ins, skip = _upsert_news_batch(session, batch)
            total_inserted += ins
            total_skipped  += skip
        except Exception as exc:
            logger.error(
                "DB upsert failed for news batch (query=%r, size=%d): %s",
                query,
                len(batch),
                exc,
            )
            session.rollback()
            summary["errors"] += 1

    summary["inserted"] = total_inserted
    summary["skipped"]  = total_skipped

    logger.info(
        "News ingestion complete (query=%r) — fetched=%d, inserted=%d, skipped=%d.",
        query,
        summary["fetched"],
        summary["inserted"],
        summary["skipped"],
    )
    return summary


def get_active_companies(session: Session) -> list[Company]:
    """
    Return all active Company records from the database.
    Used by CLI scripts to iterate over the full watchlist.
    """
    return session.query(Company).filter(Company.is_active.is_(True)).all()
