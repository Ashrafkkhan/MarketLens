"""
backend/api/schemas.py — MarketLens Phase 5
============================================
Pydantic v2 data models and serialization schemas for REST endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Payload for full end-to-end event orchestration."""
    headline: str = Field(..., description="Financial event news headline", min_length=5)
    body: Optional[str] = Field(default="", description="Optional news article body or press release text")
    ticker: Optional[str] = Field(default=None, description="Optional NSE ticker override (e.g. 'INFY', 'TCS')")
    top_k: Optional[int] = Field(default=20, description="Number of historical FAISS analogs to query", ge=1, le=100)

    model_config = {
        "json_schema_extra": {
            "example": {
                "headline": "TCS bags mega $1.2B deal from European retail giant, revenue visibility strengthens",
                "body": "Tata Consultancy Services announced a multi-year partnership to transform digital core operations.",
                "ticker": "TCS",
                "top_k": 20,
            }
        }
    }


class PredictRequest(BaseModel):
    """Payload for lightweight Phase 3 ML prediction."""
    headline: str = Field(..., description="Financial event headline")
    body: Optional[str] = Field(default="", description="Optional article body")
    ticker: Optional[str] = Field(default=None, description="NSE ticker symbol")
    event_type: Optional[str] = Field(default=None, description="Optional explicit event type")
    surprise: Optional[str] = Field(default=None, description="BEAT / MISS / IN_LINE / NOT_APPLICABLE")
    magnitude: Optional[float] = Field(default=None, description="Numeric magnitude if applicable")

    model_config = {
        "json_schema_extra": {
            "example": {
                "headline": "Infosys reports 12% profit growth, beating analyst forecasts",
                "ticker": "INFY",
                "event_type": "EARNINGS",
                "surprise": "BEAT",
            }
        }
    }


class EvidenceRequest(BaseModel):
    """Payload for lightweight Phase 4 FAISS analog retrieval."""
    query: str = Field(..., description="Query headline or description for semantic search")
    ticker: Optional[str] = Field(default=None, description="Filter analogs to this ticker")
    event_type: Optional[str] = Field(default=None, description="Filter analogs to this event type")
    surprise: Optional[str] = Field(default=None, description="Surprise category for formatting")
    top_k: int = Field(default=20, description="Number of nearest neighbors to retrieve", ge=1, le=100)

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Reserve Bank of India keeps repo rate unchanged at 6.5%",
                "event_type": "MONETARY_POLICY",
                "top_k": 20,
            }
        }
    }


# ---------------------------------------------------------------------------
# Component Response Schemas
# ---------------------------------------------------------------------------

class ExtractedEventResponse(BaseModel):
    news_id: Optional[int] = None
    ticker: Optional[str] = None
    event_type: str
    gemini_sentiment: str
    magnitude: Optional[float] = None
    surprise: Optional[str] = None
    summary: str
    finbert_positive: float
    finbert_negative: float
    finbert_neutral: float


class TechnicalsResponse(BaseModel):
    momentum_5d: float
    momentum_20d: float
    volatility_20d: float
    volume_anomaly: float
    market_return_5d: float


class PredictResponse(BaseModel):
    ticker: Optional[str] = None
    probability: float
    signal: int
    conviction: str
    target_horizon: str
    features: Dict[str, float]


class AnalogItem(BaseModel):
    rank: int
    event_id: int
    ticker: Optional[str] = None
    event_type: str = "OTHER"
    surprise: Optional[str] = "IN_LINE"
    event_date: str
    ar_3: Optional[float] = None
    similarity_score: float
    summary: str


class EvidenceResponse(BaseModel):
    candidates_fetched: int
    sample_size: int
    win_rate_ar3: Optional[float] = None
    median_ar3: Optional[float] = None
    mean_ar3: Optional[float] = None
    std_ar3: Optional[float] = None
    max_gain: Optional[float] = None
    max_loss: Optional[float] = None
    top_analogs: List[AnalogItem] = []


class SynthesisResponse(BaseModel):
    source_llm: str
    full_memo: str
    sections: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Composite Response Schemas
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    """Complete unified payload returned by /api/v1/analyze."""
    headline: str
    body: Optional[str] = ""
    extracted_event: ExtractedEventResponse
    technicals: TechnicalsResponse
    prediction: PredictResponse
    evidence: EvidenceResponse
    synthesis: SynthesisResponse


class HealthResponse(BaseModel):
    """System diagnostic and subsystem health status."""
    status: str
    database: bool
    faiss_index: bool
    model_c: bool
    details: Dict[str, Any]


# ---------------------------------------------------------------------------
# Chat & Feed Schemas (Phase 6)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Active event analysis context")


class ChatResponse(BaseModel):
    response: str
    source_llm: str


class EventFeedItem(BaseModel):
    id: int
    news_id: int
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    headline: str
    body: Optional[str] = None
    event_type: str
    sentiment: str
    surprise: Optional[str] = None
    magnitude: Optional[float] = None
    summary: str
    published_at: Optional[str] = None
    created_at: Optional[str] = None


class EventFeedResponse(BaseModel):
    events: List[EventFeedItem]
    total: int


# ---------------------------------------------------------------------------
# Price Forecast Schemas (Stock Price Prediction ML)
# ---------------------------------------------------------------------------

class PriceForecastRequest(BaseModel):
    """Payload for stock price forecast via local ML pipeline."""
    ticker: str = Field(..., description="NSE ticker symbol without suffix (e.g. 'RELIANCE', 'TCS')")
    horizon_days: int = Field(default=7, description="Number of future trading days to forecast", ge=1, le=30)
    model_name: Optional[str] = Field(default=None, description="Model to use: 'Random Forest', 'XGBoost', 'LSTM'. Auto-selects if None.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "ticker": "RELIANCE",
                "horizon_days": 7,
                "model_name": "Random Forest",
            }
        }
    }


class PriceForecastResponse(BaseModel):
    """Price forecast result from the local ML pipeline."""
    ticker: str
    model_used: Optional[str] = None
    last_actual_date: Optional[str] = None
    last_actual_close: Optional[float] = None
    forecast_dates: List[str] = []
    forecast_prices: List[float] = []
    signal: str = "Hold"
    confidence_pct: int = 0
    price_change_pct: float = 0.0
    model_metrics: Dict[str, Any] = {}
    available_models: List[str] = []
    error: Optional[str] = None
