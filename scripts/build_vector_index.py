"""
scripts/build_vector_index.py — MarketLens Phase 4
===================================================
CLI script to batch-embed all historical events from PostgreSQL and persist
the FAISS index and metadata to disk.

Workflow:
  1. Load all events (with ticker) from the `events` table via SQLAlchemy.
  2. Join with Phase 3 dataset_builder to compute AR_1, AR_3, AR_5 for each event.
  3. Format embedding strings: "[{ticker}] {event_type} | Surprise: {surprise} | {summary}"
  4. Encode via sentence-transformers/all-MiniLM-L6-v2 in batches.
  5. Build FAISS IndexFlatIP (dim=384) and persist:
       data/faiss_index.bin
       data/faiss_metadata.parquet

Usage:
    python scripts/build_vector_index.py
    python scripts/build_vector_index.py --include-no-ticker   # include macro events
    python scripts/build_vector_index.py --batch-size 32       # adjust batch size
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from database.connection import SessionLocal
from backend.ml.dataset_builder import (
    build_event_matrix,
    _load_market_data,
    _load_index_data,
)
from backend.memory.embedder import EventEmbedder, EMBEDDING_DIM
from backend.memory.vector_store import FAISSVectorStore, DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_vector_index")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_events_with_ar(include_no_ticker: bool = False) -> pd.DataFrame:
    """
    Load events from DB and augment with AR values from Phase 3 logic.
    Returns a DataFrame with all METADATA_COLS populated.
    """
    logger.info("Connecting to PostgreSQL...")
    with SessionLocal() as session:
        # Phase 3 dataset_builder gives us events with computed AR values
        event_df = build_event_matrix(session)
        mkt_df   = _load_market_data(session)
        idx_df   = _load_index_data(session)

    logger.info("Event matrix: %d rows with AR data", len(event_df))

    if event_df.empty:
        logger.warning("No events with matching market data found in DB.")
        return event_df

    if not include_no_ticker:
        before = len(event_df)
        event_df = event_df[event_df["ticker"].notna()].copy()
        logger.info("Filtered to ticker-linked events: %d (excluded %d macro)", len(event_df), before - len(event_df))

    return event_df


def build_metadata_rows(event_df: pd.DataFrame, formatted_texts: list[str]) -> list[dict]:
    """
    Build the list of metadata dicts aligned with embedding rows.
    Each dict maps to a row in faiss_metadata.parquet.
    """
    rows = []
    for i, (_, row) in enumerate(event_df.iterrows()):
        evt_date = row.get("event_date")
        rows.append({
            "faiss_id":       i,
            "event_id":       int(row.get("event_id", -1)),
            "news_id":        int(row.get("news_id", -1)),
            "ticker":         row.get("ticker"),
            "event_type":     str(row.get("event_type", "OTHER")),
            "surprise":       row.get("surprise"),
            "event_date":     pd.Timestamp(evt_date) if evt_date is not None else None,
            "summary":        str(row.get("summary", "")),
            "formatted_text": formatted_texts[i],
            "ar_1":           float(row["ar_1"]) if pd.notna(row.get("ar_1")) else None,
            "ar_3":           float(row["ar_3"]) if pd.notna(row.get("ar_3")) else None,
            "ar_5":           float(row["ar_5"]) if pd.notna(row.get("ar_5")) else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MarketLens Phase 4 — Build FAISS vector index from historical events"
    )
    parser.add_argument(
        "--include-no-ticker",
        action="store_true",
        help="Include macro/economy-wide events without a ticker (default: excluded)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help=f"Output FAISS index path (default: {DEFAULT_INDEX_PATH})",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Output metadata parquet path (default: {DEFAULT_METADATA_PATH})",
    )
    args = parser.parse_args()

    print("\n" + "=" * 74)
    print("  MARKETLENS PHASE 4 — FAISS VECTOR INDEX BUILDER")
    print("=" * 74)

    t_start = time.time()

    # 1. Load events with AR values
    logger.info("Step 1/4: Loading events with abnormal return data...")
    event_df = load_events_with_ar(include_no_ticker=args.include_no_ticker)

    if event_df.empty:
        logger.error(
            "No events found! Ensure you have:\n"
            "  1. Run scripts/extract_events.py to populate the events table.\n"
            "  2. Run scripts/collect_market_data.py to populate price data.\n"
            "  3. Ensure news dates overlap with market data date range."
        )
        sys.exit(1)

    logger.info("Loaded %d events for indexing.", len(event_df))

    # 2. Initialise embedder
    logger.info("Step 2/4: Loading embedding model '%s'...", args.model)
    embedder = EventEmbedder(model_name=args.model, batch_size=args.batch_size)

    # 3. Build embedding strings and encode
    logger.info("Step 3/4: Encoding %d events...", len(event_df))
    records = event_df[["ticker", "event_type", "surprise", "summary"]].to_dict("records")

    t_embed = time.time()
    embeddings, formatted_texts = embedder.embed_events(records, show_progress=True)
    embed_time = time.time() - t_embed
    logger.info(
        "Encoded %d events in %.1fs (%.1f events/sec). Shape: %s",
        len(formatted_texts), embed_time, len(formatted_texts) / max(embed_time, 0.001),
        embeddings.shape,
    )

    # 4. Build metadata rows
    logger.info("Step 4/4: Building FAISS index and persisting to disk...")
    metadata_rows = build_metadata_rows(event_df, formatted_texts)

    store = FAISSVectorStore(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        dim=EMBEDDING_DIM,
    )
    store.build_and_save(embeddings, metadata_rows)

    total_time = time.time() - t_start
    ar3_coverage = event_df["ar_3"].notna().sum()

    print("\n" + "=" * 74)
    print(f"  BUILD COMPLETE in {total_time:.1f}s")
    print(f"  Vectors indexed   : {store.size}")
    print(f"  AR3 coverage      : {ar3_coverage}/{len(event_df)} events have realised AR3")
    print(f"  FAISS index saved : {args.index_path}")
    print(f"  Metadata saved    : {args.metadata_path}")
    print(f"\n  Next step: python scripts/test_evidence_engine.py")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
