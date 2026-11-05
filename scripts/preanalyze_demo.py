"""
scripts/preanalyze_demo.py — pre-analyse the demo headline and save it to the cache.

Run once before a demo; after that POST /api/v1/analyze returns this headline
instantly, even after a server restart.

Usage:
    python scripts/preanalyze_demo.py
    python scripts/preanalyze_demo.py --headline "..." --ticker TCS
    python scripts/preanalyze_demo.py --force      # re-run even if already cached
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from backend.api.schemas import AnalyzeResponse
from backend.intelligence import analysis_cache
from backend.intelligence.finalpipeline import get_agent

DEFAULT_HEADLINE = "Infosys Stock Update: Shares Slip Amid IT Sector Headwinds - LatestLY"
DEFAULT_TICKER = "INFY"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline", default=DEFAULT_HEADLINE)
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    key = analysis_cache.make_key(args.headline, "", args.ticker)
    if not args.force and analysis_cache.get_cached(key):
        print(f"Already cached: {args.headline!r} ({args.ticker}). Use --force to redo.")
        return

    print(f"Analysing {args.headline!r} ({args.ticker}) — this can take a while...")
    start = time.time()
    result = get_agent().analyze_event(headline=args.headline, body="", ticker=args.ticker, top_k=20)
    response = AnalyzeResponse(**result)
    analysis_cache.save_cached(key, response.model_dump(mode="json"))
    print(f"Saved to {analysis_cache.CACHE_PATH} in {time.time() - start:.0f}s.")
    print(f"  prob={response.prediction.probability:.3f}  memo via {response.synthesis.source_llm}")


if __name__ == "__main__":
    main()
