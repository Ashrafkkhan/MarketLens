"""
backend/memory/evidence_engine.py — MarketLens Phase 4
=======================================================
Historical Evidence Aggregator.

Given a query event (text + structured attributes), retrieves the K most
semantically similar historical events via FAISS and computes empirical
market outcome statistics from their realised abnormal returns (AR_1, AR_3, AR_5).

This gives a quantitative "base-rate" answer to: "What typically happened in the
market after similar historical events?"

Key outputs per query:
  - sample_size      : total matched analogs
  - win_rate_ar3     : fraction where AR_3 > +0.5% (positive edge)
  - median_ar3       : median 3-day abnormal return across analogs
  - mean_ar3         : average 3-day abnormal return
  - std_ar3          : cross-sectional standard deviation of AR_3
  - max_gain         : best historical AR_3 outcome in matched set
  - max_loss         : worst historical AR_3 outcome in matched set
  - top_analogs      : Top 3 closest matches with date, ticker, AR_3
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.memory.embedder import EventEmbedder, get_embedder
from backend.memory.vector_store import FAISSVectorStore, get_vector_store

# ---------------------------------------------------------------------------
# Output Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AnalogRecord:
    """A single retrieved historical analog event."""
    rank: int
    event_id: int
    ticker: Optional[str]
    event_type: str
    surprise: Optional[str]
    event_date: str
    ar_3: Optional[float]
    summary: str
    similarity_score: float


@dataclass
class EvidenceReport:
    """
    Aggregated empirical evidence from historical analogs for a query event.
    """
    # Query metadata
    query_text: str
    query_ticker: Optional[str]
    query_event_type: Optional[str]
    query_surprise: Optional[str]

    # Retrieval stats
    candidates_fetched: int
    sample_size: int          # after filtering out self-matches and NaN AR_3

    # Empirical AR_3 distribution
    win_rate_ar3: float       # fraction where AR_3 > 0.005
    median_ar3: float
    mean_ar3: float
    std_ar3: float
    max_gain: float
    max_loss: float

    # AR_1 and AR_5 for cross-horizon perspective
    median_ar1: Optional[float]
    median_ar5: Optional[float]

    # Top closest analogs
    top_analogs: List[AnalogRecord] = field(default_factory=list)

    # Raw retrieved DataFrame for downstream use
    raw_matches: Optional[pd.DataFrame] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------

class EvidenceEngine:
    """
    Historical Evidence Engine: wraps EventEmbedder + FAISSVectorStore to
    answer "what happened historically after similar events?" queries.
    """

    AR3_POSITIVE_THRESHOLD = 0.005   # +0.5% vs NIFTY 50 = "positive edge"
    TOP_ANALOGS_N          = 3

    def __init__(
        self,
        embedder: Optional[EventEmbedder] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        top_k: int = 20,
    ) -> None:
        self.embedder     = embedder or get_embedder()
        self.store        = vector_store or get_vector_store()
        self.default_topk = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        event_type: Optional[str] = None,
        surprise: Optional[str] = None,
        top_k: Optional[int] = None,
        ticker_filter: Optional[str] = None,
        event_type_filter: Optional[str] = None,
        exclude_event_ids: Optional[List[int]] = None,
    ) -> EvidenceReport:
        """
        Run a semantic similarity search for historical analogs of the query event
        and return an EvidenceReport with aggregated market outcome statistics.

        Parameters
        ----------
        query_text : str
            The event headline or summary text to query against.
        ticker : str, optional
            The ticker for this query event (used in formatting, not filtering).
        event_type : str, optional
            Event classification label for richer embedding (e.g. "EARNINGS").
        surprise : str, optional
            Surprise category label (e.g. "BEAT", "MISS").
        top_k : int, optional
            Override default number of analogs to retrieve.
        ticker_filter : str, optional
            Only return historical analogs for this specific ticker.
        event_type_filter : str, optional
            Only return historical analogs of this event type.
        exclude_event_ids : list[int], optional
            Event IDs to exclude from results (prevents self-matching).

        Returns
        -------
        EvidenceReport
        """
        k = top_k or self.default_topk

        if not self.store.is_loaded:
            raise RuntimeError(
                "Vector store not loaded. "
                "Run: python scripts/build_vector_index.py"
            )

        # 1. Encode query
        query_vec = self.embedder.embed_query(
            query_text=query_text,
            ticker=ticker,
            event_type=event_type,
            surprise=surprise,
        )

        # 2. FAISS search with optional filters
        matches = self.store.search(
            query_vector=query_vec,
            top_k=k,
            ticker_filter=ticker_filter,
            event_type_filter=event_type_filter,
            exclude_event_ids=exclude_event_ids,
        )

        candidates_fetched = len(matches)

        # 3. Filter to events with usable AR_3 values
        ar3_col = matches["ar_3"].dropna()
        valid_matches = matches[matches["ar_3"].notna()].copy()
        sample_size = len(valid_matches)

        if sample_size == 0:
            logger.warning(
                "No historical analogs with AR_3 data found for query: '%s'",
                query_text[:80],
            )
            return EvidenceReport(
                query_text=query_text,
                query_ticker=ticker,
                query_event_type=event_type,
                query_surprise=surprise,
                candidates_fetched=candidates_fetched,
                sample_size=0,
                win_rate_ar3=float("nan"),
                median_ar3=float("nan"),
                mean_ar3=float("nan"),
                std_ar3=float("nan"),
                max_gain=float("nan"),
                max_loss=float("nan"),
                median_ar1=None,
                median_ar5=None,
                top_analogs=[],
                raw_matches=matches,
            )

        ar3_vals = valid_matches["ar_3"].astype(float).values

        # 4. Compute empirical distribution
        win_rate  = float(np.mean(ar3_vals > self.AR3_POSITIVE_THRESHOLD))
        median_ar3 = float(np.median(ar3_vals))
        mean_ar3   = float(np.mean(ar3_vals))
        std_ar3    = float(np.std(ar3_vals, ddof=1)) if len(ar3_vals) > 1 else 0.0
        max_gain   = float(np.max(ar3_vals))
        max_loss   = float(np.min(ar3_vals))

        # Cross-horizon (AR_1, AR_5) — best-effort
        median_ar1 = (
            float(valid_matches["ar_1"].dropna().median())
            if "ar_1" in valid_matches.columns and valid_matches["ar_1"].notna().any()
            else None
        )
        median_ar5 = (
            float(valid_matches["ar_5"].dropna().median())
            if "ar_5" in valid_matches.columns and valid_matches["ar_5"].notna().any()
            else None
        )

        # 5. Build top-analog records (closest N)
        top_n = valid_matches.head(self.TOP_ANALOGS_N)
        top_analogs = []
        for rank, (_, row) in enumerate(top_n.iterrows(), start=1):
            evt_date = row.get("event_date")
            if hasattr(evt_date, "strftime"):
                evt_date_str = evt_date.strftime("%Y-%m-%d")
            else:
                evt_date_str = str(evt_date)[:10]

            top_analogs.append(AnalogRecord(
                rank=rank,
                event_id=int(row.get("event_id", -1)),
                ticker=row.get("ticker"),
                event_type=str(row.get("event_type", "")),
                surprise=row.get("surprise"),
                event_date=evt_date_str,
                ar_3=float(row["ar_3"]) if pd.notna(row.get("ar_3")) else None,
                summary=str(row.get("summary", ""))[:200],
                similarity_score=float(row.get("similarity_score", 0.0)),
            ))

        return EvidenceReport(
            query_text=query_text,
            query_ticker=ticker,
            query_event_type=event_type,
            query_surprise=surprise,
            candidates_fetched=candidates_fetched,
            sample_size=sample_size,
            win_rate_ar3=win_rate,
            median_ar3=median_ar3,
            mean_ar3=mean_ar3,
            std_ar3=std_ar3,
            max_gain=max_gain,
            max_loss=max_loss,
            median_ar1=median_ar1,
            median_ar5=median_ar5,
            top_analogs=top_analogs,
            raw_matches=valid_matches,
        )

    # ------------------------------------------------------------------
    # Display helper
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: EvidenceReport) -> None:
        """Pretty-print a structured EvidenceReport to stdout."""
        border = "=" * 74
        sub    = "-" * 74

        print(f"\n{border}")
        print(f"  MARKETLENS — HISTORICAL EVIDENCE ENGINE REPORT")
        print(f"{border}")
        print(f"  Query    : {report.query_text[:72]}")
        print(f"  Ticker   : {report.query_ticker or 'MACRO'}")
        print(f"  Type     : {report.query_event_type or 'N/A'} | "
              f"Surprise: {report.query_surprise or 'N/A'}")
        print(f"{sub}")
        print(f"  Historical Sample Statistics (AR3 = Abnormal Return vs NIFTY50, 3-day)")
        print(f"{sub}")

        if report.sample_size == 0:
            print("  ⚠️  No historical analogs with AR3 data found in the index.")
        else:
            print(f"  Candidates Fetched : {report.candidates_fetched}")
            print(f"  Sample Size (AR3)  : {report.sample_size}")
            print(f"  Win Rate  (AR3>0.5%): {report.win_rate_ar3*100:.1f}%")
            print(f"  Median AR3         : {report.median_ar3*100:+.3f}%")
            print(f"  Mean AR3           : {report.mean_ar3*100:+.3f}%")
            print(f"  Std Dev AR3        : {report.std_ar3*100:.3f}%")
            print(f"  Max Gain  (AR3)    : {report.max_gain*100:+.3f}%")
            print(f"  Max Loss  (AR3)    : {report.max_loss*100:+.3f}%")
            if report.median_ar1 is not None:
                print(f"  Median AR1 (1-day) : {report.median_ar1*100:+.3f}%")
            if report.median_ar5 is not None:
                print(f"  Median AR5 (5-day) : {report.median_ar5*100:+.3f}%")

            print(f"\n  Top {len(report.top_analogs)} Historical Analogues:")
            print(f"  {'#':<4} {'Sim':>6} {'Ticker':<12} {'EventType':<22} {'Date':<12} {'AR3':>8}  Summary")
            print(f"  {'-'*4} {'-'*6} {'-'*12} {'-'*22} {'-'*12} {'-'*8}  {'-'*30}")
            for a in report.top_analogs:
                sim_str  = f"{a.similarity_score:.3f}"
                ticker   = (a.ticker or "MACRO")[:12]
                et       = (a.event_type or "")[:22]
                ar3_str  = f"{a.ar_3*100:+.3f}%" if a.ar_3 is not None else "   N/A"
                summary  = (a.summary or "")[:45]
                print(f"  #{a.rank:<3} {sim_str:>6} {ticker:<12} {et:<22} {a.event_date:<12} {ar3_str:>8}  {summary}")

        print(f"{border}\n")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_default_engine: Optional[EvidenceEngine] = None


def get_evidence_engine(top_k: int = 20) -> EvidenceEngine:
    """Return the cached EvidenceEngine (lazy-init, loads vector store on first call)."""
    global _default_engine
    if _default_engine is None:
        _default_engine = EvidenceEngine(top_k=top_k)
    return _default_engine
