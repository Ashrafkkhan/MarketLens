"""
backend/api/routes.py — MarketLens Phase 5
===========================================
FastAPI route definitions for event analysis, ML prediction, vector evidence,
and system health checks.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import text

from database.connection import SessionLocal
from database.models import Event, News, Company
from backend.intelligence.finalpipeline import MarketLensAgent, get_agent
from backend.memory.vector_store import DEFAULT_INDEX_PATH, DEFAULT_METADATA_PATH
from backend.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EventFeedItem,
    EventFeedResponse,
    EvidenceRequest,
    EvidenceResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PriceForecastRequest,
    PriceForecastResponse,
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
def analyze_event(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        from backend.intelligence import analysis_cache

        cache_key = analysis_cache.make_key(payload.headline, payload.body, payload.ticker)
        cached = analysis_cache.get_cached(cache_key)
        if cached:
            try:
                return AnalyzeResponse(**cached)
            except Exception as ce:
                logger.warning("Ignoring invalid cached analysis: %s", ce)

        agent: MarketLensAgent = get_agent()
        result = agent.analyze_event(
            headline=payload.headline,
            body=payload.body or "",
            ticker=payload.ticker,
            top_k=payload.top_k or 20,
        )
        response = AnalyzeResponse(**result)
        try:
            analysis_cache.save_cached(cache_key, response.model_dump(mode="json"))
        except Exception as se:
            logger.warning("Could not save analysis to cache: %s", se)
        return response
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
def predict_event(payload: PredictRequest) -> PredictResponse:
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
def get_evidence(payload: EvidenceRequest) -> EvidenceResponse:
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
                "ticker": str(a.ticker) if (pd.notna(a.ticker) and a.ticker) else None,
                "event_type": str(a.event_type) if (pd.notna(a.event_type) and a.event_type) else "OTHER",
                "surprise": str(a.surprise) if (pd.notna(a.surprise) and a.surprise) else "IN_LINE",
                "event_date": str(a.event_date) if pd.notna(a.event_date) else "",
                "ar_3": float(a.ar_3) if pd.notna(a.ar_3) else None,
                "similarity_score": float(a.similarity_score) if pd.notna(a.similarity_score) else 0.0,
                "summary": str(a.summary) if pd.notna(a.summary) else "",
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
def health_check() -> HealthResponse:
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


@router.get(
    "/events",
    response_model=EventFeedResponse,
    summary="Fetch Recent Extracted Financial Events",
    description="Returns the latest ingested news events with structured categories, sentiment, and ticker tags.",
)
def get_recent_events(limit: int = 40) -> EventFeedResponse:
    try:
        with SessionLocal() as session:
            rows = (
                session.query(Event, News, Company.company_name)
                .join(News, Event.news_id == News.id)
                .outerjoin(Company, Event.ticker == Company.ticker)
                .order_by(Event.created_at.desc())
                .limit(limit)
                .all()
            )

            event_items = []
            for evt, news, comp_name in rows:
                event_items.append(
                    EventFeedItem(
                        id=evt.id,
                        news_id=evt.news_id,
                        ticker=evt.ticker,
                        company_name=comp_name,
                        headline=news.headline,
                        body=news.article,
                        event_type=evt.event_type,
                        sentiment=evt.gemini_sentiment,
                        surprise=evt.surprise,
                        magnitude=evt.magnitude,
                        summary=evt.summary,
                        published_at=news.published_at.isoformat() if news.published_at else None,
                        created_at=evt.created_at.isoformat() if evt.created_at else None,
                    )
                )

            return EventFeedResponse(events=event_items, total=len(event_items))
    except Exception as e:
        logger.exception("Error in /events endpoint: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch events: {str(e)}",
        )


@router.post(
    "/price-forecast",
    response_model=PriceForecastResponse,
    summary="Stock Price Forecast (Local ML Pipeline)",
    description=(
        "Runs the local Random Forest / XGBoost / LSTM price prediction pipeline "
        "to forecast future closing prices for an NSE ticker. "
        "100%% local — no Gemini, no cloud API. "
        "Requires pre-trained model artefacts in ml_genai/stock-price-prediction-ml/saved_models/."
    ),
)
def price_forecast(payload: PriceForecastRequest) -> PriceForecastResponse:
    try:
        from backend.services.price_forecast_service import run_price_forecast

        result = run_price_forecast(
            ticker=payload.ticker,
            horizon_days=payload.horizon_days,
            model_name=payload.model_name,
        )

        return PriceForecastResponse(**result)
    except Exception as e:
        logger.exception("Error in /price-forecast endpoint: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Price forecast error: {str(e)}",
        )


FORECAST_BACKTEST_NOTE = (
    "Backtest (RELIANCE, 60 rolling origins, 7 trading days): model mean MAPE 1.81% vs naive "
    "'tomorrow = today' 1.62%; day-7 direction correct 55%. The forecast does not beat the current price."
)

_PRICE_INTENT_KEYWORDS = (
    "price", "predict", "prediction", "forecast", "invest", "target",
    "worth", "buy", "sell", "valuation", "outlook",
)


def _resolve_ticker_from_query(user_query: str) -> Optional[str]:
    """
    Scan *user_query* for an explicit NSE ticker symbol or company name match
    against the active companies table. Returns the base ticker, or None if
    nothing in the message names a specific company.
    """
    try:
        with SessionLocal() as session:
            rows = (
                session.query(Company.ticker, Company.company_name)
                .filter(Company.is_active.is_(True))
                .all()
            )
    except Exception as e:
        logger.debug("Ticker resolution DB lookup failed: %s", e)
        return None

    q_upper = user_query.upper()

    # 1. Exact ticker symbol as a whole word (e.g. "TCS", "RELIANCE")
    for ticker, _ in rows:
        if ticker and re.search(rf"\b{re.escape(ticker)}\b", q_upper):
            return ticker

    # 2. Company name mention (strip common corporate suffixes), longest match wins
    q_lower = user_query.lower()
    best_ticker, best_len = None, 0
    for ticker, name in rows:
        if not name:
            continue
        core = re.sub(r"\b(ltd|limited|inc|corp|corporation|plc|company|co)\b\.?", "", name.lower()).strip()
        if core and len(core) > 2 and core in q_lower and len(core) > best_len:
            best_ticker, best_len = ticker, len(core)

    return best_ticker


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Interactive Analyst Chat Assistant",
    description="Grounded conversational assistant answering equity research questions on the active event using Gemini or Ollama.",
)
def chat_with_analyst(payload: ChatRequest) -> ChatResponse:
    try:
        user_query = ""
        for m in reversed(payload.messages):
            if m.role == "user" and m.content and m.content.strip():
                user_query = m.content.strip()
                break

        if not user_query:
            return ChatResponse(
                response="Hello! I am your MarketLens quantitative copilot. How can I assist you with analyzing the active event, Model C probabilities, or historical analogs?",
                source_llm="deterministic_analyst_copilot",
            )

        ctx = payload.context or {}
        headline = ctx.get("headline", "No active headline")
        ext = ctx.get("extracted_event", {})
        ticker = ext.get("ticker", "MACRO")
        evt_type = ext.get("event_type", "OTHER")
        surprise = ext.get("surprise", "N/A")
        sentiment = ext.get("sentiment") or ext.get("gemini_sentiment", "NEUTRAL")
        pred = ctx.get("prediction", {})
        prob = pred.get("probability", 0.5) * 100
        signal = pred.get("signal", 0)
        ev = ctx.get("evidence", {})
        sample_size = ev.get("sample_size", 0)
        win_rate = ev.get("win_rate_ar3")
        win_str = f"{win_rate * 100:.1f}%" if win_rate is not None else "N/A"
        med_ar3 = ev.get("median_ar3")
        med_str = f"{med_ar3 * 100:+.2f}%" if med_ar3 is not None else "N/A"
        top_analogs = ev.get("top_analogs", [])
        analogs_summary = "; ".join([
            f"[{a.get('event_date')}] {a.get('ticker')} (AR3: {a.get('ar_3', 0)*100:+.1f}%)"
            for a in top_analogs[:3]
        ]) or "None"

        # Price forecast context from local ML pipeline. Resolve the ticker the
        # user is actually asking about (explicit mention > active cockpit event),
        # and fetch a live, ticker-correct forecast rather than trusting whatever
        # forecast (possibly for a different ticker) the frontend last attached.
        from backend.services.price_forecast_service import (
            run_price_forecast,
            get_training_status,
            start_training_if_needed,
        )

        fc = ctx.get("price_forecast", {}) or {}
        wants_price = any(k in user_query.lower() for k in _PRICE_INTENT_KEYWORDS)
        mentioned_ticker = _resolve_ticker_from_query(user_query)
        effective_ticker = mentioned_ticker or (ticker if ticker and ticker != "MACRO" else None)

        if wants_price:
            if not effective_ticker:
                return ChatResponse(
                    response="Which stock would you like me to forecast? Name a ticker or company (e.g. 'RELIANCE', 'TCS', 'Infosys') and I'll run the local price model on it.",
                    source_llm="deterministic_analyst_copilot",
                )

            fc_live = run_price_forecast(ticker=effective_ticker, horizon_days=7)

            if fc_live.get("error") and "No pre-trained models" in fc_live["error"]:
                status = get_training_status(effective_ticker)
                if status and status.get("status") == "training":
                    elapsed = int(time.time() - status["started_at"])
                    return ChatResponse(
                        response=f"Still training the local Random Forest / XGBoost / LSTM models for {effective_ticker} on 2021-present data (started {elapsed}s ago). This usually takes a few minutes on CPU — ask me again shortly.",
                        source_llm="deterministic_analyst_copilot",
                    )
                if status and status.get("status") == "failed":
                    return ChatResponse(
                        response=f"Training failed for {effective_ticker}: {status.get('error')}. Please confirm it's a valid NSE ticker and try again.",
                        source_llm="deterministic_analyst_copilot",
                    )
                start_training_if_needed(effective_ticker)
                return ChatResponse(
                    response=f"No pre-trained price model exists for {effective_ticker} yet. I've started training local Random Forest, XGBoost, and LSTM models on 2021-present historical data in the background — this typically takes a few minutes. Ask me again shortly and I'll have a predicted price ready.",
                    source_llm="deterministic_analyst_copilot",
                )

            if not fc_live.get("error"):
                fc = fc_live

        fc_section = ""
        if fc and not fc.get("error"):
            fc_signal = fc.get("signal", "Hold")
            fc_conf = fc.get("confidence_pct", 0)
            fc_change = fc.get("price_change_pct", 0.0)
            fc_model = fc.get("model_used", "ML Model")
            fc_last_close = fc.get("last_actual_close")
            fc_prices = fc.get("forecast_prices", [])
            fc_dates = fc.get("forecast_dates", [])
            fc_metrics = fc.get("model_metrics", {})
            fc_mape = fc_metrics.get("MAPE", None)
            fc_r2 = fc_metrics.get("R2", None)

            if fc_prices and fc_dates:
                first_pred = fc_prices[0]
                last_pred = fc_prices[-1]
                fc_ticker = fc.get("ticker", effective_ticker or ticker)
                fc_section = (
                    f"\n\n## PRICE FORECAST for {fc_ticker} (Local {fc_model} ML Model):\n"
                    f"- Last Known Close: ₹{fc_last_close:,.2f} (as of {fc.get('last_actual_date', 'N/A')})\n"
                    f"- Day +1 Forecast: ₹{first_pred:,.2f}\n"
                    f"- Day +{len(fc_prices)} Forecast: ₹{last_pred:,.2f}\n"
                    f"- Predicted Price Change: {fc_change:+.2f}% over {len(fc_prices)} trading days\n"
                    f"- Day-1 vs last close: {fc_signal} (a direction hint, not a recommendation)\n"
                    f"- {FORECAST_BACKTEST_NOTE if (fc_ticker == 'RELIANCE' and fc_model == 'Random Forest') else 'No backtest exists for this ticker/model; treat its accuracy as unknown.'}\n"
                )
                if fc_mape is not None:
                    fc_section += f"- Test-set MAPE={fc_mape:.2f}% (one-step, not multi-day accuracy)"
                if fc_r2 is not None:
                    fc_section += f", R²={fc_r2:.4f}\n"

        system_instruction = f"""You are MarketLens Copilot, an institutional quantitative equity research assistant for Indian stock markets.
You are assisting a portfolio manager examining the active event analysis:
- Active Event Headline: "{headline}"
- Ticker: {ticker}
- Classification: {evt_type} | Surprise: {surprise} | Sentiment: {sentiment}
- Model C Probability of AR3 > +0.5%: {prob:.1f}% (Signal: {signal})
- FAISS Historical Analogs: Sample Size={sample_size}, Win Rate={win_str}, Median AR3={med_str}
- Top Historical Precedents: {analogs_summary}{fc_section}

CAPABILITIES & GUARDRAILS:
1. Natural Conversation: Greet the user normally and answer general questions conversationally, politely, and professionally.
2. Active Event Questions: Answer questions about the active event ({ticker}) using the exact Model C probabilities, win rates, analogs, and risks above.
3. Price Forecasts: When the user asks whether to buy or sell, or for the predicted price, answer directly. Start your reply with one line in the form "VERDICT: BUY <TICKER>", "VERDICT: SELL <TICKER>", or "VERDICT: HOLD <TICKER>", where you decide the verdict yourself from the expected change in the PRICE FORECAST section: BUY for a clear rise, SELL for a clear fall, HOLD when the change is small. Then, in one short paragraph, give the predicted price, the expected change, and the direction over the forecast days, and add one sentence with the backtest result from the forecast section. Do NOT add disclaimers, risk warnings, or general advice. Do NOT use the words "consult", "risk", "past performance", "may not", or "important to note". Do NOT describe any figure as a confidence or probability.
4. New Headlines / Other Stocks: If the user provides a new headline or asks to evaluate another stock (like Infosys, Reliance, etc.), provide an intuitive analyst perspective and politely let them know they can run our full quantitative ML pipeline on it by using the "+ Analyze" button at the top of the Live Feed on the left!
5. Zero Hallucination: Do not invent false numerical return percentages or fake price targets outside the model data."""

        import os

        # Tier 0: Groq cloud (fast, OpenAI-compatible). Falls through on any failure.
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                import requests as _rq

                groq_msgs = [{"role": "system", "content": system_instruction}]
                for m in payload.messages:
                    if m.content and m.content.strip() and m.role in ("user", "assistant"):
                        groq_msgs.append({"role": m.role, "content": m.content.strip()})
                groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
                gres = _rq.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    json={"model": groq_model, "messages": groq_msgs, "temperature": 0.3, "max_tokens": 800, "reasoning_effort": "low"},
                    timeout=8.0,
                )
                if gres.status_code == 200:
                    gans = (gres.json()["choices"][0]["message"].get("content") or "").strip()
                    if gans:
                        return ChatResponse(response=gans, source_llm=f"groq/{groq_model}")
                else:
                    logger.warning("Groq chat HTTP %s: %s", gres.status_code, gres.text[:200])
            except Exception as ge:
                logger.warning("Groq chat failed, falling back: %s", ge)

        # Tier 1: Try Google Gemini Chat API (recommended AFC-safe approach)
        api_key =os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=10000))
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                )

                # Build conversation history (all turns except the last user message)
                history = []
                for m in payload.messages[:-1]:
                    if m.content and m.content.strip():
                        role = "model" if m.role == "assistant" else "user"
                        history.append(
                            types.Content(
                                role=role,
                                parts=[types.Part(text=m.content.strip())]
                            )
                        )

                model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
                # Only use models confirmed available via API listing
                for m in [model_name, "gemini-3.8-flash", "gemini-flash-latest"]:
                    try:
                        chat = client.chats.create(
                            model=m,
                            history=history,
                            config=config,
                        )
                        res = chat.send_message(user_query)
                        if res and res.text:
                            return ChatResponse(response=res.text.strip(), source_llm=f"google/{m}")
                    except Exception as me:
                        logger.debug("Chat attempt with %s failed: %s", m, me)
            except Exception as e:
                logger.warning("Gemini chat failed, falling back: %s", e)

        # Tier 2: Try Ollama
        import requests
        try:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            res = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_query},
                    ],
                    "stream": False,
                },
                timeout=4.0,
            )
            if res.status_code == 200:
                answer = res.json().get("message", {}).get("content", "").strip()
                if answer:
                    return ChatResponse(response=answer, source_llm="ollama/qwen2.5:1.5b")
        except Exception:
            pass

        # Tier 2.5: Try Local HuggingFace Qwen2.5-1.5B-Instruct Model
        try:
            import backend.services.event_extractor as ee
            import torch

            ee._load_qwen()
            tokenizer = ee._qwen_tokenizer
            model = ee._qwen_model
            device = ee._qwen_device

            if tokenizer is not None and model is not None:
                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_query},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([text], return_tensors="pt").to(device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.9,
                    )
                new_tokens = out[0][inputs.input_ids.shape[-1]:]
                ans = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                if ans:
                    return ChatResponse(response=ans, source_llm="local/Qwen2.5-1.5B-Instruct")
        except Exception as qe:
            logger.debug("Local Qwen chat generation failed: %s", qe)

        # Tier 3: Deterministic Analyst Copilot Fallback
        q_lower = user_query.lower()
        if "win rate" in q_lower or "historical" in q_lower or "rate" in q_lower:
            fallback = f"The historical win rate across {sample_size} FAISS analogs is {win_str} with a median realized 3-day abnormal return of {med_str}. The top closest matches were: {analogs_summary}."
        elif "volume" in q_lower or "technicals" in q_lower:
            fallback = f"Pre-event volume anomaly is {ctx.get('technicals', {}).get('volume_anomaly', 1.0):.2f}x of the 20-day average. A volume surge confirms institutional participation; low volume indicates headline noise."
        elif "model" in q_lower or "probability" in q_lower or "conviction" in q_lower:
            fallback = f"Model C forecasts a {prob:.1f}% probability of 3-day abnormal return outperformance (> +0.5% vs NIFTY 50). The binary signal is {signal} ({'BUY/OUTPERFORM' if signal == 1 else 'HOLD/NEUTRAL'})."
        elif "drawdown" in q_lower or "risk" in q_lower or "worst" in q_lower:
            fallback = f"The maximum realized loss observed across the historical analogs was {ev.get('max_loss', 0)*100:+.2f}%. An invalidation trigger occurs if the stock breaks -1.0% vs the index on T+1."
        else:
            fallback = f"Based on the active event for {ticker} ({evt_type}), Model C assigns a {prob:.1f}% probability of outperformance over T+3. FAISS historical analogs show a {win_str} win rate across {sample_size} matches."

        return ChatResponse(response=fallback, source_llm="deterministic_analyst_copilot")
    except Exception as e:
        logger.exception("Error in /chat endpoint: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat assistant error: {str(e)}",
        )

