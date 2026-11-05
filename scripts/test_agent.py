"""
scripts/test_agent.py — MarketLens Phase 5
===========================================
CLI Verification Script for the End-to-End Orchestration Agent.

Executes the unified pipeline:
  1. Event Extraction (Phase 2)
  2. Technical Market Alignment (Phase 3)
  3. ML Model C Inference (Phase 3)
  4. FAISS Historical Precedents (Phase 4)
  5. Multi-factor LLM Synthesis (Phase 5)

Usage:
    python scripts/test_agent.py
    python scripts/test_agent.py --ticker TCS --headline "TCS secures major multi-year IT contract"
    python scripts/test_agent.py --headline "Infosys beats Q2 profit estimates by 8%, raises FY guidance" --ticker INFY
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Quiet internal loggers so the final report prints cleanly
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

from backend.intelligence.finalpipeline import MarketLensAgent, get_agent


DEFAULT_HEADLINE = (
    "TCS bags mega $1.2B deal from European retail giant, revenue visibility strengthens"
)
DEFAULT_BODY = (
    "Tata Consultancy Services (TCS) today announced it has signed a strategic $1.2 billion "
    "multi-year contract with a premier European retail enterprise. The engagement encompasses "
    "cloud migration, AI-driven supply chain automation, and digital customer channels over five years."
)
DEFAULT_TICKER = "TCS"


def print_banner(title: str, char: str = "=") -> None:
    width = 78
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def print_report_card(result: dict) -> None:
    """Format and print the institutional research memo card to stdout."""
    ext = result["extracted_event"]
    tech = result["technicals"]
    pred = result["prediction"]
    ev = result["evidence"]
    synth = result["synthesis"]

    print_banner("MARKETLENS INTELLIGENCE ENGINE — END-TO-END AGENT REPORT")
    print(f"  Headline : {result['headline']}")
    if result.get("body"):
        print(f"  Body     : {result['body'][:75]}...")
    print(f"  Ticker   : {ext.get('ticker') or 'MACRO'}")

    # Section 1: Extraction & Sentiment
    print_banner("1. STRUCTURED EXTRACTION & NLP SENTIMENT (Phase 2)", "-")
    print(f"  Event Type : {ext.get('event_type')} | Surprise: {ext.get('surprise') or 'N/A'}")
    print(f"  Sentiment  : {ext.get('gemini_sentiment')} | Magnitude: {ext.get('magnitude') or 'N/A'}")
    print(f"  Summary    : {ext.get('summary')}")
    print(
        f"  FinBERT    : Positive={ext.get('finbert_positive', 0)*100:.1f}% | "
        f"Negative={ext.get('finbert_negative', 0)*100:.1f}% | "
        f"Neutral={ext.get('finbert_neutral', 0)*100:.1f}%"
    )

    # Section 2: Technicals
    print_banner("2. TECHNICAL & MARKET CONTEXT (Phase 3 Inputs)", "-")
    print(f"  5-Day Momentum   : {tech.get('momentum_5d', 0)*100:+.2f}%")
    print(f"  20-Day Momentum  : {tech.get('momentum_20d', 0)*100:+.2f}%")
    print(f"  20-Day Volatility: {tech.get('volatility_20d', 0)*100:.2f}%")
    print(f"  Volume Anomaly   : {tech.get('volume_anomaly', 1.0):.2f}x of 20-day mean")
    print(f"  NIFTY 50 5-Day   : {tech.get('market_return_5d', 0)*100:+.2f}%")

    # Section 3: ML Forecast
    print_banner("3. QUANTITATIVE MODEL C PREDICTION (Phase 3)", "-")
    prob = pred.get("probability", 0.5) * 100
    signal_str = "BUY / OUTPERFORM (> +0.5% AR3)" if pred.get("signal") == 1 else "HOLD / NEUTRAL"
    print(f"  Outperformance Prob : {prob:.1f}%")
    print(f"  Model Signal        : {pred.get('signal')} ({signal_str})")
    print(f"  Conviction Level    : {pred.get('conviction')}")
    print(f"  Target Horizon      : {pred.get('target_horizon')}")

    # Section 4: FAISS Historical Evidence
    print_banner("4. HISTORICAL ANALOG EVIDENCE (Phase 4 — FAISS)", "-")
    n_sample = ev.get("sample_size", 0)
    win_rate = ev.get("win_rate_ar3")
    win_str = f"{win_rate*100:.1f}%" if win_rate is not None and not (isinstance(win_rate, float) and win_rate != win_rate) else "N/A"
    med_ar3 = ev.get("median_ar3")
    med_str = f"{med_ar3*100:+.3f}%" if med_ar3 is not None and not (isinstance(med_ar3, float) and med_ar3 != med_ar3) else "N/A"
    max_g = ev.get("max_gain")
    max_l = ev.get("max_loss")
    max_g_str = f"{max_g*100:+.2f}%" if max_g is not None and not (isinstance(max_g, float) and max_g != max_g) else "N/A"
    max_l_str = f"{max_l*100:+.2f}%" if max_l is not None and not (isinstance(max_l, float) and max_l != max_l) else "N/A"

    print(f"  Analogs Retrieved : {n_sample}")
    print(f"  Historical Win Rate: {win_str} (Realized AR3 > +0.5%)")
    print(f"  Median 3-Day AR3  : {med_str}")
    print(f"  Realized Range    : Max Gain {max_g_str} | Max Loss {max_l_str}")

    top_analogs = ev.get("top_analogs", [])
    if top_analogs:
        print("\n  Top Retrieved Historical Analogues:")
        print(f"  {'#':<3} {'Sim':>5} {'Ticker':<10} {'Date':<12} {'Realized AR3':>14}  Summary")
        print(f"  {'-'*3} {'-'*5} {'-'*10} {'-'*12} {'-'*14}  {'-'*26}")
        for a in top_analogs[:3]:
            a_score = a.get("similarity_score", 0)
            a_ticker = (a.get("ticker") or "MACRO")[:10]
            a_date = str(a.get("event_date"))[:10]
            a_ret = a.get("ar_3")
            a_ret_str = f"{a_ret*100:+.2f}%" if a_ret is not None else "N/A"
            a_sum = (a.get("summary") or "")[:40]
            print(f"  #{a.get('rank', 1):<2} {a_score:>5.3f} {a_ticker:<10} {a_date:<12} {a_ret_str:>14}  {a_sum}")

    # Section 5: Synthesized Memo
    print_banner(f"5. SYNTHESIZED RESEARCH MEMO (Phase 5 — {synth.get('source_llm')})", "-")
    print(synth.get("full_memo", "").strip())
    print_banner("END OF REPORT")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MarketLens Phase 5 — End-to-End GenAI Orchestration CLI"
    )
    parser.add_argument(
        "--headline",
        type=str,
        default=DEFAULT_HEADLINE,
        help=f"Event news headline (default: '{DEFAULT_HEADLINE}')",
    )
    parser.add_argument(
        "--body",
        type=str,
        default=DEFAULT_BODY,
        help="Optional article body or press release excerpt",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=DEFAULT_TICKER,
        help="Target stock ticker override (e.g. TCS, INFY)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of historical analogs to retrieve (default: 20)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("  MARKETLENS PHASE 5 — AGENTIC ORCHESTRATION PIPELINE")
    print("=" * 78)
    print(f"  Query Headline : {args.headline}")
    print(f"  Target Ticker  : {args.ticker}")
    print("  Initializing agentic pipeline...\n")

    t_start = time.time()
    agent = get_agent()
    result = agent.analyze_event(
        headline=args.headline,
        body=args.body,
        ticker=args.ticker,
        top_k=args.top_k,
    )
    duration = time.time() - t_start

    print_report_card(result)
    print(f"\n[Completed end-to-end orchestration in {duration:.2f}s]\n")


if __name__ == "__main__":
    main()
