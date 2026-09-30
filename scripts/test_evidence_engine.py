"""
scripts/test_evidence_engine.py — MarketLens Phase 4
=====================================================
CLI verification script for the Historical Evidence Engine.

Tests vector retrieval and empirical outcome aggregation against
predefined query headlines and an optional interactive mode.

Usage:
    python scripts/test_evidence_engine.py
    python scripts/test_evidence_engine.py --interactive
    python scripts/test_evidence_engine.py --top-k 15 --ticker INFY
    python scripts/test_evidence_engine.py --query "Infosys profit jumps 15% beating estimates"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.WARNING,  # Keep engine internals quiet; we control display
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("test_evidence_engine")

from backend.memory.evidence_engine import EvidenceEngine, EvidenceReport, get_evidence_engine
from backend.memory.vector_store import DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH


# ---------------------------------------------------------------------------
# Pre-defined test queries
# ---------------------------------------------------------------------------

PREDEFINED_QUERIES = [
    {
        "label": "Query 1: Earnings Beat — IT Sector",
        "text": "Infosys reports 10% jump in profit, beating market consensus",
        "ticker": "INFY",
        "event_type": "EARNINGS",
        "surprise": "BEAT",
    },
    {
        "label": "Query 2: Monetary Policy — RBI Rate Decision",
        "text": "Reserve Bank of India holds repo rate steady at 6.5%, signals cautious stance",
        "ticker": None,
        "event_type": "MONETARY_POLICY",
        "surprise": "IN_LINE",
    },
    {
        "label": "Query 3: Earnings Miss — Banking",
        "text": "HDFC Bank quarterly profit disappoints, net interest margin compresses",
        "ticker": "HDFCBANK",
        "event_type": "EARNINGS",
        "surprise": "MISS",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_index_exists() -> bool:
    """Return True if both index files exist."""
    return DEFAULT_INDEX_PATH.exists() and DEFAULT_METADATA_PATH.exists()


def run_query_and_display(
    engine: EvidenceEngine,
    query_text: str,
    ticker: str | None,
    event_type: str | None,
    surprise: str | None,
    ticker_filter: str | None,
    event_type_filter: str | None,
    top_k: int,
    label: str = "Custom Query",
) -> EvidenceReport:
    """Execute a single evidence query and print the structured report."""
    print(f"\n{'=' * 74}")
    print(f"  {label}")
    print(f"{'=' * 74}")

    report = engine.query(
        query_text=query_text,
        ticker=ticker,
        event_type=event_type,
        surprise=surprise,
        top_k=top_k,
        ticker_filter=ticker_filter,
        event_type_filter=event_type_filter,
    )

    EvidenceEngine.print_report(report)
    return report


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(engine: EvidenceEngine, top_k: int) -> None:
    """Allow the user to enter a custom query from the terminal."""
    print("\n" + "=" * 74)
    print("  MARKETLENS EVIDENCE ENGINE — INTERACTIVE MODE")
    print("=" * 74)
    print("Enter a financial event headline to retrieve historical analogs.\n")

    query_text = input("  Event Headline / Summary: ").strip()
    if not query_text:
        print("  No input provided. Exiting.")
        return

    ticker = input("  Ticker (e.g. INFY, TCS) [leave blank for macro]: ").strip() or None
    event_type = input(
        "  Event Type [EARNINGS/MONETARY_POLICY/REGULATION/CONTRACT_ORDER/"
        "\n             MERGER_ACQUISITION/MANAGEMENT_CHANGE/MACRO_DATA/OTHER] (or blank): "
    ).strip().upper() or None
    surprise = input(
        "  Surprise [BEAT/MISS/IN_LINE/NOT_APPLICABLE] (or blank): "
    ).strip().upper() or None

    run_query_and_display(
        engine=engine,
        query_text=query_text,
        ticker=ticker,
        event_type=event_type,
        surprise=surprise,
        ticker_filter=None,
        event_type_filter=None,
        top_k=top_k,
        label=f"Interactive: {query_text[:50]}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MarketLens Phase 4 — Historical Evidence Engine Tester"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single custom query headline (runs one query instead of predefined set)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker for the query event (used in embedding, not filtering)",
    )
    parser.add_argument(
        "--event-type",
        type=str,
        default=None,
        help="Event type label for the query (e.g. EARNINGS)",
    )
    parser.add_argument(
        "--surprise",
        type=str,
        default=None,
        help="Surprise category for the query (e.g. BEAT)",
    )
    parser.add_argument(
        "--ticker-filter",
        type=str,
        default=None,
        help="Restrict historical analogs to this ticker only",
    )
    parser.add_argument(
        "--event-type-filter",
        type=str,
        default=None,
        help="Restrict historical analogs to this event type only",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of nearest neighbors to retrieve from FAISS (default: 20)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal prompt for custom queries",
    )
    args = parser.parse_args()

    # 1. Check index exists
    if not check_index_exists():
        print("\n" + "=" * 74)
        print("  ERROR: FAISS index not found!")
        print("=" * 74)
        print(f"  Expected: {DEFAULT_INDEX_PATH}")
        print(f"  Expected: {DEFAULT_METADATA_PATH}")
        print(f"\n  Please run the index builder first:")
        print(f"    python scripts/build_vector_index.py")
        print("=" * 74 + "\n")
        sys.exit(1)

    print("\n" + "=" * 74)
    print("  MARKETLENS PHASE 4 — HISTORICAL EVIDENCE ENGINE TEST")
    print("=" * 74)

    # 2. Initialise engine (loads FAISS + metadata from disk)
    print("  Loading vector store and embedding model...")
    engine = get_evidence_engine(top_k=args.top_k)

    print(f"  Vector store: {engine.store.size} events indexed")
    print(f"  Top-K        : {args.top_k}")

    # 3. Route to correct mode
    if args.interactive:
        run_interactive(engine, top_k=args.top_k)

    elif args.query:
        # Single custom CLI query
        run_query_and_display(
            engine=engine,
            query_text=args.query,
            ticker=args.ticker,
            event_type=args.event_type,
            surprise=args.surprise,
            ticker_filter=args.ticker_filter,
            event_type_filter=args.event_type_filter,
            top_k=args.top_k,
            label=f"Custom: {args.query[:50]}",
        )

    else:
        # Default: run all predefined queries
        print(f"\n  Running {len(PREDEFINED_QUERIES)} pre-defined verification queries...\n")
        reports = []
        for q in PREDEFINED_QUERIES:
            report = run_query_and_display(
                engine=engine,
                query_text=q["text"],
                ticker=q.get("ticker"),
                event_type=q.get("event_type"),
                surprise=q.get("surprise"),
                ticker_filter=args.ticker_filter,
                event_type_filter=args.event_type_filter,
                top_k=args.top_k,
                label=q["label"],
            )
            reports.append(report)

        # Summary table
        print("=" * 74)
        print("  VERIFICATION SUMMARY")
        print("=" * 74)
        print(f"  {'Query':<40} {'N':>4} {'WinRate':>8} {'MedianAR3':>10}")
        print(f"  {'-'*40} {'-'*4} {'-'*8} {'-'*10}")
        for q, r in zip(PREDEFINED_QUERIES, reports):
            label_short = q["label"][:40]
            if r.sample_size > 0:
                win_str    = f"{r.win_rate_ar3*100:.1f}%"
                median_str = f"{r.median_ar3*100:+.3f}%"
            else:
                win_str    = "  N/A"
                median_str = "       N/A"
            print(f"  {label_short:<40} {r.sample_size:>4} {win_str:>8} {median_str:>10}")
        print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
