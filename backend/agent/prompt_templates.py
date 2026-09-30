"""
backend/agent/prompt_templates.py — MarketLens Phase 5
======================================================
Strict system prompts, guardrails, and templates for institutional equity research synthesis.

Guarantees:
  - Strict Hallucination Barrier: Only use provided ML model metrics and FAISS historical stats.
  - Zero Made-Up Data: Do not fabricate price targets, imaginary corporate dates, or fake returns.
  - Institutional Tone: Concise, hypothesis-driven, quantitative tone suitable for buy-side desks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

SYNTHESIS_SYSTEM_PROMPT = """\
You are an Executive Quantitative Equity Research Analyst at an institutional asset management firm.
Your mandate is to synthesize multi-modal event intelligence, technical market momentum, statistical machine learning predictions, and historical FAISS analogs into an institutional Event Briefing Memo.

CRITICAL GUARDRAILS & HALLUCINATION BARRIER:
1. STRICT TRUTH HORIZON: Strictly adhere to the numbers, probabilities, win rates, and historical analogs provided in the input payload.
2. ZERO FABRICATION: Never invent price targets, specific rupee target values, unmentioned dates, or fake historical precedents.
3. CLEAR LABELING: Always distinguish between the Model C statistical probability (e.g., 68.4% probability of AR3 > +0.5%) and empirical historical analogs (e.g., historical win rate 60.0% across N matches).
4. QUANTITATIVE TONE: Be succinct, objective, and analytical. Avoid generic marketing adjectives.

Your memo MUST follow this exact 4-part structure:

### 1. Executive Thesis
- Conviction Rating: [High Conviction Bullish | Moderate Bullish | Neutral | Moderate Bearish | High Conviction Bearish]
- Statistical Edge: State the ML Model C probability of 3-day abnormal return outperformance (> +0.5% vs NIFTY 50) and target signal.
- Core Thesis: 2-3 sentences synthesizing the fundamental news catalyst with the quantified probability.

### 2. Technical & Event Catalysts
- Explain the key driving features that influenced the model:
  - Event Categorization: Event Type, Surprise category, Magnitude.
  - Sentiment Breakdown: FinBERT positive/negative/neutral distribution and NLP sentiment.
  - Technical Baseline: 5-day/20-day momentum, 20-day volatility, volume anomaly ratio, and NIFTY 50 5-day market backdrop.

### 3. Historical Analog Precedents
- Empirical Base Rate: Cite FAISS sample size, historical win rate (AR3 > +0.5%), median AR3, and return spread (Max Gain / Max Loss).
- Case Studies: Reference the top retrieved historical analog events (date, ticker, event summary, and realized AR3 outcome).
- Base-Rate vs Model Alignment: Note whether the historical precedent aligns with or diverges from the ML model's prediction.

### 4. Downside Risks & Invalidation Triggers
- Structural & Execution Risks: Specific operational or sector risks associated with this event.
- Invalidation Triggers: Quantitative market condition or technical price level that would invalidate this thesis (e.g., benchmark underperformance, volume exhaustion).
"""

USER_SYNTHESIS_TEMPLATE = """\
Analyze the following event payload and synthesize an Institutional Event Briefing Memo following the 4-part structure.

---
[INPUT PAYLOAD]

1. NEWS CATALYST:
- Headline: {headline}
- Body: {body}
- Assigned Ticker: {ticker}

2. STRUCTURED EXTRACTION (Phase 2):
- Event Type: {event_type}
- Sentiment: {sentiment}
- Surprise: {surprise}
- Magnitude: {magnitude}
- Core Summary: {summary}
- FinBERT Sentiment: Positive={finbert_pos:.3f}, Negative={finbert_neg:.3f}, Neutral={finbert_neu:.3f}

3. TECHNICAL BACKDROP (Phase 3 Inputs):
- 5-Day Stock Momentum: {momentum_5d}
- 20-Day Stock Momentum: {momentum_20d}
- 20-Day Realized Volatility: {volatility_20d}
- Volume Anomaly Ratio: {volume_anomaly}x
- NIFTY 50 5-Day Return: {market_return_5d}

4. QUANTITATIVE MODEL C INFERENCE (Phase 3):
- Target Definition: 3-Day Abnormal Return vs NIFTY 50 > +0.5% (AR3 > 0.005)
- Model Outperformance Probability: {probability:.1f}%
- Binary Signal: {signal} ({signal_label})
- Conviction: {conviction}

5. HISTORICAL FAISS ANALOG EVIDENCE (Phase 4):
- Semantic Analogs Fetched: {sample_size}
- Historical Win Rate (AR3 > +0.5%): {win_rate_str}
- Median Historical AR3: {median_ar3_str}
- Average Historical AR3: {mean_ar3_str}
- Realized Range: Max Gain {max_gain_str} | Max Loss {max_loss_str}
- Top Historical Analog Precedents:
{top_analogs_str}
---

Generate the complete institutional memo now:"""


def format_synthesis_prompt(
    headline: str,
    body: str,
    extracted_event: Dict[str, Any],
    technicals: Dict[str, Any],
    prediction: Dict[str, Any],
    evidence: Dict[str, Any],
) -> str:
    """Format the full quantitative context into the strict user prompt."""
    ticker = extracted_event.get("ticker") or "MACRO / UNASSIGNED"
    event_type = extracted_event.get("event_type", "OTHER")
    sentiment = extracted_event.get("gemini_sentiment") or extracted_event.get("sentiment", "NEUTRAL")
    surprise = extracted_event.get("surprise") or "NOT_APPLICABLE"
    mag_val = extracted_event.get("magnitude")
    magnitude = f"{mag_val}" if mag_val is not None else "N/A"
    summary = extracted_event.get("summary", "")

    finbert_pos = float(extracted_event.get("finbert_positive", 0.0))
    finbert_neg = float(extracted_event.get("finbert_negative", 0.0))
    finbert_neu = float(extracted_event.get("finbert_neutral", 0.0))

    # Technicals formatting
    def fmt_pct(val: Optional[float]) -> str:
        if val is None:
            return "N/A"
        return f"{val * 100:+.2f}%"

    momentum_5d = fmt_pct(technicals.get("momentum_5d"))
    momentum_20d = fmt_pct(technicals.get("momentum_20d"))
    vol_val = technicals.get("volatility_20d")
    volatility_20d = f"{vol_val * 100:.2f}%" if vol_val is not None else "N/A"
    vol_anom = technicals.get("volume_anomaly")
    volume_anomaly = f"{vol_anom:.2f}" if vol_anom is not None else "1.00"
    market_return_5d = fmt_pct(technicals.get("market_return_5d"))

    # Prediction formatting
    probability = float(prediction.get("probability", 0.5)) * 100.0
    signal = int(prediction.get("signal", 1 if probability >= 50.0 else 0))
    signal_label = "OUTPERFORM (AR3 > +0.5%)" if signal == 1 else "NO OUTPERFORMANCE"
    conviction = prediction.get("conviction", "MODERATE")

    # Evidence formatting
    sample_size = evidence.get("sample_size", 0)
    win_rate = evidence.get("win_rate_ar3")
    win_rate_str = f"{win_rate * 100:.1f}%" if win_rate is not None and not (isinstance(win_rate, float) and (win_rate != win_rate)) else "N/A"
    
    med_ar3 = evidence.get("median_ar3")
    median_ar3_str = fmt_pct(med_ar3) if med_ar3 is not None and not (isinstance(med_ar3, float) and (med_ar3 != med_ar3)) else "N/A"
    
    mean_ar3 = evidence.get("mean_ar3")
    mean_ar3_str = fmt_pct(mean_ar3) if mean_ar3 is not None and not (isinstance(mean_ar3, float) and (mean_ar3 != mean_ar3)) else "N/A"
    
    max_gain = evidence.get("max_gain")
    max_gain_str = fmt_pct(max_gain) if max_gain is not None and not (isinstance(max_gain, float) and (max_gain != max_gain)) else "N/A"
    
    max_loss = evidence.get("max_loss")
    max_loss_str = fmt_pct(max_loss) if max_loss is not None and not (isinstance(max_loss, float) and (max_loss != max_loss)) else "N/A"

    # Top analogs
    top_analogs = evidence.get("top_analogs", [])
    if top_analogs:
        analog_lines = []
        for a in top_analogs[:3]:
            if isinstance(a, dict):
                a_ticker = a.get("ticker", "MACRO")
                a_date = a.get("event_date", "Unknown Date")
                a_ar3 = a.get("ar_3")
                a_ar3_str = fmt_pct(a_ar3) if a_ar3 is not None else "N/A"
                a_sum = a.get("summary", "")[:70]
                a_score = a.get("similarity_score", 0.0)
            else:
                a_ticker = getattr(a, "ticker", "MACRO")
                a_date = getattr(a, "event_date", "Unknown Date")
                a_ar3 = getattr(a, "ar_3", None)
                a_ar3_str = fmt_pct(a_ar3) if a_ar3 is not None else "N/A"
                a_sum = getattr(a, "summary", "")[:70]
                a_score = getattr(a, "similarity_score", 0.0)
            analog_lines.append(
                f"  - [{a_date}] {a_ticker} (Cosine Sim: {a_score:.3f}): Realized AR3 = {a_ar3_str}. \"{a_sum}\""
            )
        top_analogs_str = "\n".join(analog_lines)
    else:
        top_analogs_str = "  - No historical analogs available in vector store."

    clean_body = body.strip() if body and body.strip() else "None provided."

    return USER_SYNTHESIS_TEMPLATE.format(
        headline=headline,
        body=clean_body,
        ticker=ticker,
        event_type=event_type,
        sentiment=sentiment,
        surprise=surprise,
        magnitude=magnitude,
        summary=summary,
        finbert_pos=finbert_pos,
        finbert_neg=finbert_neg,
        finbert_neu=finbert_neu,
        momentum_5d=momentum_5d,
        momentum_20d=momentum_20d,
        volatility_20d=volatility_20d,
        volume_anomaly=volume_anomaly,
        market_return_5d=market_return_5d,
        probability=probability,
        signal=signal,
        signal_label=signal_label,
        conviction=conviction,
        sample_size=sample_size,
        win_rate_str=win_rate_str,
        median_ar3_str=median_ar3_str,
        mean_ar3_str=mean_ar3_str,
        max_gain_str=max_gain_str,
        max_loss_str=max_loss_str,
        top_analogs_str=top_analogs_str,
    )
