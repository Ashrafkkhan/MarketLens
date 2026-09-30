"""
backend/api/routes.py — MarketLens Phase 5
===========================================
FastAPI route definitions for event analysis, ML prediction, vector evidence,
and system health checks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text

from database.connection import SessionLocal
from backend.agent.orchestrator import MarketLensAgent, get_agent
from backend.memory.vector_store import DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH
from backend.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    EvidenceRequest,
    EvidenceResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["MarketLens Intelligence Engine"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="End-to-End Event Analysis & Research Synthesis",
    description=(
        "Executes the full 5-stage orchestration pipeline: "
        "1) Local event extraction & FinBERT scoring, "
        "2) Technical feature alignment from DB, "
        "3) Phase 3 Model C inference, "
        "4) Phase 4 FAISS historical analog search, and "
        "5) Multi-factor LLM research memo synthesis."
    ),
)
async def analyze_event(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        agent: MarketLensAgent = get_agent()
        result = agent.analyze_event(
            headline=payload.headline,
            body=payload.body or "",
            ticker=payload.ticker,
            top_k=payload.top_k or 20,
        )
        return AnalyzeResponse(**result)
    except Exception as e:
        logger.exception("Error in /analyze endpoint: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {str(e)}",
        )


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Phase 3 ML Abnormal Return Prediction Only",
    description="Lightweight endpoint computing Model C 3-day abnormal return probability and classification signal.",
)
async def predict_event(payload: PredictRequest) -> PredictResponse:
    try:
        agent: MarketLensAgent = get_agent()
        with SessionLocal() as session:
            # 1. Extraction / sentiment
            extracted = agent.extractor_service.extract(
                news_id=0,
                headline=payload.headline,
                article_body=payload.body,
                session=session,
            )
            # Apply user overrides
            if payload.ticker:
                extracted["ticker"] = payload.ticker.split(".")[0].strip().upper()
            if payload.event_type:
                extracted["event_type"] = payload.event_type.upper()
            if payload.surprise:
                extracted["surprise"] = payload.surprise.upper()
            if payload.magnitude is not None:
                extracted["magnitude"] = payload.magnitude

            # 2. Technicals from DB
            technicals = agent.compute_technicals_from_db(
                ticker=extracted.get("ticker"),
                session=session,
            )

        # 3. Model inference
        prediction = agent.predict_event(
            extracted_event=extracted,
            technicals=technicals,
        )
        return PredictResponse(**prediction)
    except Exception as e:
        logger.exception("Error in /predict endpoint: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


@router.post(
    "/evidence",
    response_model=EvidenceResponse,
    summary="Phase 4 FAISS Semantic Evidence Retrieval",
    description="Retrieves Top-K historical event analogs and empirical abnormal return distributions.",
)
async def get_evidence(payload: EvidenceRequest) -> EvidenceResponse:
    try:
        agent: MarketLensAgent = get_agent()
        report = agent.evidence_engine.query(
            query_text=payload.query,
            ticker=payload.ticker,
            event_type=payload.event_type,
            surprise=payload.surprise,
            top_k=payload.top_k,
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
            for a in report.top_analogs
        ]

        return EvidenceResponse(
            candidates_fetched=report.candidates_fetched,
            sample_size=report.sample_size,
            win_rate_ar3=report.win_rate_ar3,
            median_ar3=report.median_ar3,
            mean_ar3=report.mean_ar3,
            std_ar3=report.std_ar3,
            max_gain=report.max_gain,
            max_loss=report.max_loss,
            top_analogs=top_analogs_data,
        )
    except Exception as e:
        logger.exception("Error in /evidence endpoint: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evidence retrieval error: {str(e)}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Diagnostic & Health Check",
    description="Verifies database connectivity, FAISS vector index readiness, and Model C artifact presence.",
)
async def health_check() -> HealthResponse:
    db_ok = False
    faiss_ok = False
    model_ok = False
    details: Dict[str, Any] = {}

    # Check Database
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
            db_ok = True
            details["database"] = "PostgreSQL connected and responsive"
    except Exception as e:
        details["database"] = f"DB connection failed: {str(e)}"

    # Check FAISS index
    try:
        if DEFAULT_INDEX_PATH.exists() and DEFAULT_METADATA_PATH.exists():
            agent = get_agent()
            size = agent.evidence_engine.store.size
            faiss_ok = True
            details["faiss_index"] = f"IndexFlatIP ready with {size} historical events"
        else:
            details["faiss_index"] = f"Index files missing at {DEFAULT_INDEX_PATH}"
    except Exception as e:
        details["faiss_index"] = f"FAISS check error: {str(e)}"

    # Check Model C
    try:
        agent = get_agent()
        if agent._pipeline is not None:
            model_ok = True
            details["model_c"] = f"Operational with {len(agent._feature_cols)} features"
        else:
            details["model_c"] = f"Model artifact missing or failed to load at {agent.model_path}"
    except Exception as e:
        details["model_c"] = f"Model check error: {str(e)}"

    overall_status = "HEALTHY" if (db_ok and faiss_ok and model_ok) else "DEGRADED"

    return HealthResponse(
        status=overall_status,
        database=db_ok,
        faiss_index=faiss_ok,
        model_c=model_ok,
        details=details,
    )
