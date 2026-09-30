"""
backend/agent/orchestrator.py — MarketLens Phase 5
===================================================
Central agent controller executing the end-to-end quantitative pipeline:
  1. Extraction (Phase 2): Event classification & FinBERT sentiment via local NLP.
  2. Technicals & Feature Engineering (Phase 3): Recent price/volume alignment from DB.
  3. ML Model C Inference (Phase 3): 3-day abnormal return probability & signal.
  4. FAISS Semantic Evidence (Phase 4): Vector retrieval of historical precedents.
  5. Multi-factor LLM Synthesis (Phase 5): Institutional research memo generation.
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

# Suppress minor sklearn version warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import SessionLocal
from database.models import Company, MarketData, MarketIndex
from backend.services.event_extractor import EventExtractorService
from backend.memory.evidence_engine import EvidenceEngine, get_evidence_engine
from backend.agent.synthesizer import ReportSynthesizer, get_synthesizer

logger = logging.getLogger(__name__)

# Default model artifact path
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "event_classifier_v1.joblib"

EVENT_TYPES = [
    "EARNINGS", "MONETARY_POLICY", "REGULATION", "CONTRACT_ORDER",
    "MERGER_ACQUISITION", "MANAGEMENT_CHANGE", "MACRO_DATA", "OTHER",
]
SURPRISE_TYPES = ["BEAT", "MISS", "IN_LINE", "NOT_APPLICABLE"]


class MarketLensAgent:
    """
    Unified Orchestration Agent for real-time equity event intelligence.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        evidence_engine: Optional[EvidenceEngine] = None,
        synthesizer: Optional[ReportSynthesizer] = None,
        extractor_service: Optional[EventExtractorService] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.evidence_engine = evidence_engine or get_evidence_engine()
        self.synthesizer = synthesizer or get_synthesizer()
        self.extractor_service = extractor_service or EventExtractorService()
        
        # Pipeline artifact state
        self._pipeline = None
        self._feature_cols: List[str] = []
        self._model_metadata: Dict[str, Any] = {}
        self._load_pipeline()

    # ------------------------------------------------------------------
    # Pipeline loading
    # ------------------------------------------------------------------
    def _load_pipeline(self) -> None:
        """Load and introspect Model C pipeline artifact."""
        if not self.model_path.exists():
            logger.warning("Model C artifact not found at %s. ML inference will use heuristic fallback.", self.model_path)
            return

        try:
            raw_artifact = joblib.load(self.model_path)
            if isinstance(raw_artifact, dict):
                self._pipeline = raw_artifact.get("pipeline") or raw_artifact.get("model")
                self._feature_cols = (
                    raw_artifact.get("feature_cols")
                    or raw_artifact.get("features")
                    or raw_artifact.get("columns")
                    or []
                )
                self._model_metadata = {
                    k: v for k, v in raw_artifact.items() if k not in ["pipeline", "model"]
                }
            else:
                self._pipeline = raw_artifact

            if not self._feature_cols and hasattr(self._pipeline, "feature_names_in_"):
                self._feature_cols = list(self._pipeline.feature_names_in_)

            logger.info("MarketLensAgent loaded Model C with %d features.", len(self._feature_cols))
        except Exception as e:
            logger.error("Failed loading Model C artifact: %s", e)

    # ------------------------------------------------------------------
    # Step 2: Database Technical Feature Retrieval
    # ------------------------------------------------------------------
    def compute_technicals_from_db(
        self,
        ticker: Optional[str],
        session: Session,
    ) -> Dict[str, float]:
        """
        Query latest 30 trading days of market data for ticker and NIFTY 50
        to compute realistic momentum, volatility, and volume anomaly.
        """
        defaults = {
            "momentum_5d": 0.0,
            "momentum_20d": 0.0,
            "volatility_20d": 0.015,
            "volume_anomaly": 1.0,
            "market_return_5d": 0.0,
        }

        if not ticker:
            return defaults

        base_ticker = ticker.split(".")[0].strip().upper()

        # Query recent stock data
        stock_rows = (
            session.query(MarketData.timestamp, MarketData.close, MarketData.volume)
            .filter(MarketData.ticker == base_ticker)
            .order_by(MarketData.timestamp.desc())
            .limit(35)
            .all()
        )

        # Query recent index data
        index_rows = (
            session.query(MarketIndex.timestamp, MarketIndex.close)
            .filter(MarketIndex.index_symbol == "^NSEI")
            .order_by(MarketIndex.timestamp.desc())
            .limit(35)
            .all()
        )

        if not stock_rows or len(stock_rows) < 5:
            return defaults

        # Sort ascending in time
        stock_df = pd.DataFrame(stock_rows, columns=["timestamp", "close", "volume"])
        stock_df = stock_df.sort_values("timestamp").reset_index(drop=True)
        closes = stock_df["close"].astype(float).values
        volumes = stock_df["volume"].astype(float).values

        n = len(closes)
        mom5 = (closes[-1] - closes[-5]) / closes[-5] if n >= 5 and closes[-5] > 0 else 0.0
        mom20 = (closes[-1] - closes[-20]) / closes[-20] if n >= 20 and closes[-20] > 0 else mom5

        # 20-day volatility
        pct_changes = pd.Series(closes).pct_change().dropna()
        recent_changes = pct_changes.tail(20)
        vol20 = float(recent_changes.std(ddof=1)) if len(recent_changes) > 2 else 0.015
        if np.isnan(vol20) or vol20 == 0:
            vol20 = 0.015

        # Volume anomaly: latest volume / 20-day mean volume
        recent_vol = volumes[-20:] if n >= 20 else volumes
        avg_vol = float(np.mean(recent_vol)) if len(recent_vol) > 0 else 0.0
        vol_anomaly = float(volumes[-1] / avg_vol) if avg_vol > 0 else 1.0

        # Market return 5d
        market_ret_5d = 0.0
        if index_rows and len(index_rows) >= 5:
            idx_df = pd.DataFrame(index_rows, columns=["timestamp", "close"]).sort_values("timestamp")
            idx_closes = idx_df["close"].astype(float).values
            if len(idx_closes) >= 5 and idx_closes[-5] > 0:
                market_ret_5d = float((idx_closes[-1] - idx_closes[-5]) / idx_closes[-5])

        return {
            "momentum_5d": float(mom5),
            "momentum_20d": float(mom20),
            "volatility_20d": float(vol20),
            "volume_anomaly": float(vol_anomaly),
            "market_return_5d": float(market_ret_5d),
        }

    # ------------------------------------------------------------------
    # Step 3: ML Feature Alignment & Inference
    # ------------------------------------------------------------------
    def predict_event(
        self,
        extracted_event: Dict[str, Any],
        technicals: Dict[str, float],
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Build 21-feature row and run Model C inference.
        """
        row_data: Dict[str, float] = {}

        # Technicals
        row_data["momentum_5d"] = float(technicals.get("momentum_5d", 0.0))
        row_data["momentum_20d"] = float(technicals.get("momentum_20d", 0.0))
        row_data["volatility_20d"] = float(technicals.get("volatility_20d", 0.015))
        row_data["volume_anomaly"] = float(technicals.get("volume_anomaly", 1.0))
        row_data["market_return_5d"] = float(technicals.get("market_return_5d", 0.0))

        # Sentiments
        row_data["finbert_positive"] = float(extracted_event.get("finbert_positive", 0.0))
        row_data["finbert_negative"] = float(extracted_event.get("finbert_negative", 0.0))
        row_data["finbert_neutral"] = float(extracted_event.get("finbert_neutral", 0.0))

        # Categoricals - Event Type
        event_type = str(extracted_event.get("event_type", "OTHER")).upper()
        for et in EVENT_TYPES:
            row_data[f"evt_{et.lower()}"] = 1.0 if event_type == et else 0.0

        # Categoricals - Surprise
        surprise = str(extracted_event.get("surprise") or "NOT_APPLICABLE").upper()
        for sv in SURPRISE_TYPES:
            row_data[f"surp_{sv.lower()}"] = 1.0 if surprise == sv else 0.0

        # Event Numeric
        mag = extracted_event.get("magnitude")
        row_data["magnitude"] = float(mag) if mag is not None else 0.0

        # Construct DataFrame
        df = pd.DataFrame([row_data])

        if self._feature_cols:
            for col in self._feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self._feature_cols]

        prob = 0.50
        signal = 0

        if self._pipeline is not None:
            try:
                if hasattr(self._pipeline, "predict_proba"):
                    probas = self._pipeline.predict_proba(df)
                    prob = float(probas[0, 1])
                elif hasattr(self._pipeline, "predict"):
                    preds = self._pipeline.predict(df)
                    prob = float(preds[0])
                signal = int(prob >= threshold)
            except Exception as e:
                logger.error("Inference execution failed: %s", e)
        else:
            # High-fidelity heuristic fallback based on FinBERT + surprise
            pos = row_data["finbert_positive"]
            neg = row_data["finbert_negative"]
            is_beat = 1 if surprise == "BEAT" else 0
            prob = float(np.clip(0.40 + 0.35 * pos - 0.25 * neg + 0.15 * is_beat, 0.05, 0.95))
            signal = int(prob >= threshold)

        # Conviction label
        if prob >= 0.70:
            conviction = "HIGH CONVICTION BULLISH"
        elif prob >= 0.55:
            conviction = "MODERATE BULLISH"
        elif prob >= 0.45:
            conviction = "NEUTRAL"
        elif prob >= 0.30:
            conviction = "MODERATE BEARISH"
        else:
            conviction = "HIGH CONVICTION BEARISH"

        return {
            "ticker": extracted_event.get("ticker"),
            "probability": prob,
            "signal": signal,
            "conviction": conviction,
            "target_horizon": "T+3 Trading Days (AR3 vs NIFTY 50)",
            "features": row_data,
        }

    # ------------------------------------------------------------------
    # Main Agent Entry Point
    # ------------------------------------------------------------------
    def analyze_event(
        self,
        headline: str,
        body: str = "",
        ticker: Optional[str] = None,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """
        Execute full 5-stage sequential orchestration.
        """
        logger.info("MarketLensAgent received event: %s (Ticker override: %s)", headline[:60], ticker)

        # 1. Extraction (Phase 2)
        with SessionLocal() as session:
            extracted_event = self.extractor_service.extract(
                news_id=0,
                headline=headline,
                article_body=body,
                session=session,
            )

            # User explicit ticker override if provided
            if ticker and ticker.strip():
                clean_ticker = ticker.split(".")[0].strip().upper()
                extracted_event["ticker"] = clean_ticker

            active_ticker = extracted_event.get("ticker")

            # 2. Technical Alignment from DB
            technicals = self.compute_technicals_from_db(
                ticker=active_ticker,
                session=session,
            )

        # 3. Model C ML Inference
        prediction = self.predict_event(
            extracted_event=extracted_event,
            technicals=technicals,
        )

        # 4. Historical FAISS Evidence Retrieval (Phase 4)
        evidence_report = self.evidence_engine.query(
            query_text=headline,
            ticker=active_ticker,
            event_type=extracted_event.get("event_type"),
            surprise=extracted_event.get("surprise"),
            top_k=top_k,
        )

        top_analogs_data = [
            {
                "rank": a.rank,
                "event_id": a.event_id,
                "ticker": a.ticker,
                "event_type": a.event_type,
                "surprise": a.surprise,
                "event_date": a.event_date,
                "ar_3": a.ar_3,
                "similarity_score": a.similarity_score,
                "summary": a.summary,
            }
            for a in evidence_report.top_analogs
        ]

        evidence_dict = {
            "candidates_fetched": evidence_report.candidates_fetched,
            "sample_size": evidence_report.sample_size,
            "win_rate_ar3": evidence_report.win_rate_ar3,
            "median_ar3": evidence_report.median_ar3,
            "mean_ar3": evidence_report.mean_ar3,
            "std_ar3": evidence_report.std_ar3,
            "max_gain": evidence_report.max_gain,
            "max_loss": evidence_report.max_loss,
            "top_analogs": top_analogs_data,
        }

        # 5. Report Synthesis (Phase 5)
        synthesis = self.synthesizer.synthesize_report(
            headline=headline,
            body=body,
            extracted_event=extracted_event,
            technicals=technicals,
            prediction=prediction,
            evidence=evidence_dict,
        )

        # Unified response payload
        return {
            "headline": headline,
            "body": body,
            "extracted_event": extracted_event,
            "technicals": technicals,
            "prediction": prediction,
            "evidence": evidence_dict,
            "synthesis": synthesis,
        }


_default_agent: Optional[MarketLensAgent] = None


def get_agent() -> MarketLensAgent:
    """Return the cached MarketLensAgent singleton."""
    global _default_agent
    if _default_agent is None:
        _default_agent = MarketLensAgent()
    return _default_agent
