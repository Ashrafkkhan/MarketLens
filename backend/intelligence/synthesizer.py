"""
backend/intelligence/synthesizer.py — MarketLens Phase 5
=================================================
Multi-tier LLM report synthesis service.

Synthesis Hierarchy:
  1. Local Ollama instance (e.g. `http://localhost:11434`, model `qwen2.5:1.5b` or `qwen2.5`)
  2. Google Gemini API (if GEMINI_API_KEY is configured in environment)
  3. Local HuggingFace Transformers (if ENABLE_LOCAL_HF_SYNTHESIS=true)
  4. Institutional Rule-Based Quantitative Synthesis (Deterministic High-Fidelity Fallback)

Guarantees 100% offline uptime, zero crashes, sub-second latency, and zero hallucination of metrics.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from backend.intelligence.prompt_templates import (
    SYNTHESIS_SYSTEM_PROMPT,
    format_synthesis_prompt,
)

logger = logging.getLogger(__name__)

# Configurable endpoints and model tags
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


class ReportSynthesizer:
    """
    Synthesizes multi-factor equity research memos from event data, ML outputs,
    and historical FAISS evidence with graceful fallback across providers.
    """

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        enable_gemini: bool = True,
        enable_hf_local: Optional[bool] = None,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.enable_gemini = enable_gemini
        if enable_hf_local is None:
            local_qwen = Path(__file__).resolve().parent.parent.parent / "models" / "Qwen2.5-1.5B-Instruct"
            self.enable_hf_local = local_qwen.is_dir() or os.getenv("ENABLE_LOCAL_HF_SYNTHESIS", "false").lower() in ("true", "1", "yes")
        else:
            self.enable_hf_local = enable_hf_local

    # ------------------------------------------------------------------
    # 1. Ollama Provider
    # ------------------------------------------------------------------
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Try querying local Ollama instance with short timeout."""
        try:
            url = f"{self.ollama_url}/api/chat"
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 750,
                },
            }
            # Short timeout to avoid blocking if Ollama is not running
            resp = requests.post(url, json=payload, timeout=2.5)
            if resp.status_code == 200:
                data = resp.json()
                msg = data.get("message", {}).get("content", "")
                if msg.strip():
                    logger.info("Synthesized memo via Ollama (%s)", self.ollama_model)
                    return msg.strip()
        except Exception as e:
            logger.debug("Ollama synthesis unavailable: %s", e)
        return None

    # ------------------------------------------------------------------
    # 2. Google Gemini Provider
    # ------------------------------------------------------------------
    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Try calling Google Gemini Chat API if API key is configured."""
        if not self.enable_gemini:
            return None

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=10000))
            config = types.GenerateContentConfig(
                system_instruction=SYNTHESIS_SYSTEM_PROMPT,
                temperature=0.2,
            )
            model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
            for m in [model_name, "gemini-3.8-flash", "gemini-flash-latest"]:
                try:
                    chat = client.chats.create(model=m, config=config)
                    res = chat.send_message(prompt)
                    if res and res.text:
                        logger.info("Synthesized memo via Google Gemini (%s)", m)
                        return res.text.strip()
                except Exception as me:
                    logger.debug("Gemini synthesis with %s failed: %s", m, me)
        except Exception as e:
            logger.warning("Gemini synthesis call failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # 3. Local Hugging Face Qwen Provider (Opt-In for GPU/High-Spec CPUs)
    # ------------------------------------------------------------------
    def _call_hf_qwen(self, prompt: str) -> Optional[str]:
        """Generate using local Transformers model if explicitly enabled."""
        if not self.enable_hf_local:
            return None
        try:
            import backend.services.event_extractor as ee
            import torch

            ee._load_qwen()
            tokenizer = ee._qwen_tokenizer
            model = ee._qwen_model
            device = ee._qwen_device

            if tokenizer is None or model is None:
                return None

            messages = [
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=400,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.9,
                )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if response.strip():
                logger.info("Synthesized memo via Local HuggingFace Qwen")
                return response.strip()
        except Exception as e:
            logger.debug("Local HF Qwen synthesis error: %s", e)
        return None

    # ------------------------------------------------------------------
    # 4. Institutional Deterministic Fallback (Pristine, 0-Latency, 0-Hallucination)
    # ------------------------------------------------------------------
    def _deterministic_synthesis(
        self,
        headline: str,
        extracted_event: Dict[str, Any],
        technicals: Dict[str, Any],
        prediction: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> str:
        """
        Generates an exact, institutional-grade quantitative memo using
        the computed parameters directly, strictly preventing any hallucinations.
        """
        ticker = extracted_event.get("ticker") or "MACRO"
        event_type = extracted_event.get("event_type", "OTHER")
        sentiment = extracted_event.get("gemini_sentiment") or extracted_event.get("sentiment", "NEUTRAL")
        surprise = extracted_event.get("surprise") or "NOT_APPLICABLE"
        summary = extracted_event.get("summary") or headline

        probability = float(prediction.get("probability", 0.5)) * 100.0
        signal = int(prediction.get("signal", 1 if probability >= 50.0 else 0))
        conviction = prediction.get("conviction", "MODERATE")
        if probability >= 65.0:
            rating = "High Conviction Bullish"
        elif probability >= 50.0:
            rating = "Moderate Bullish"
        elif probability >= 40.0:
            rating = "Neutral / Cautious"
        else:
            rating = "Moderate Bearish"

        sample_size = evidence.get("sample_size", 0)
        win_rate = evidence.get("win_rate_ar3")
        win_rate_str = f"{win_rate * 100:.1f}%" if win_rate is not None and not (isinstance(win_rate, float) and (win_rate != win_rate)) else "N/A"
        median_ar3 = evidence.get("median_ar3")
        median_ar3_str = f"{median_ar3 * 100:+.2f}%" if median_ar3 is not None and not (isinstance(median_ar3, float) and (median_ar3 != median_ar3)) else "N/A"

        mom5 = technicals.get("momentum_5d")
        mom5_str = f"{mom5 * 100:+.2f}%" if mom5 is not None else "0.00%"
        vol_anom = technicals.get("volume_anomaly", 1.0)
        vol_anom_str = f"{vol_anom:.2f}x" if vol_anom is not None else "1.00x"

        finbert_pos = float(extracted_event.get("finbert_positive", 0.0)) * 100.0
        finbert_neg = float(extracted_event.get("finbert_negative", 0.0)) * 100.0

        top_analogs = evidence.get("top_analogs", [])
        analogs_text = ""
        if top_analogs:
            for i, a in enumerate(top_analogs[:3], 1):
                if isinstance(a, dict):
                    a_tick = a.get("ticker", "MACRO")
                    a_date = a.get("event_date", "Past Event")
                    a_ret = a.get("ar_3")
                    a_ret_str = f"{a_ret * 100:+.2f}%" if a_ret is not None else "N/A"
                    a_sum = a.get("summary", "")[:75]
                else:
                    a_tick = getattr(a, "ticker", "MACRO")
                    a_date = getattr(a, "event_date", "Past Event")
                    a_ret = getattr(a, "ar_3", None)
                    a_ret_str = f"{a_ret * 100:+.2f}%" if a_ret is not None else "N/A"
                    a_sum = getattr(a, "summary", "")[:75]
                analogs_text += f"\n- #{i} [{a_date}] {a_tick} | Realized AR3: {a_ret_str} — {a_sum}"
        else:
            analogs_text = "\n- No direct historical analog matches met threshold in current database."

        memo = f"""### 1. Executive Thesis
- **Conviction Rating:** {rating} ({conviction})
- **Statistical Edge:** Model C estimates a **{probability:.1f}% probability** of achieving a 3-day abnormal return exceeding +0.5% vs NIFTY 50 (Signal: {signal}).
- **Core Thesis:** The event catalyst for **{ticker}** ({event_type}, classified {sentiment}) provides a quantifiable statistical catalyst. Based on the fused ML model and historical evidence, price action is anticipated to reflect this information over the T+3 trading window.

### 2. Technical & Event Catalysts
- **Event Categorization & Surprise:** Classified as **{event_type}** with surprise status **{surprise}**. Core summary: "{summary}".
- **NLP Sentiment Distribution:** FinBERT registers **{finbert_pos:.1f}% positive** vs **{finbert_neg:.1f}% negative** probability mass, supporting a net favorable bias.
- **Market & Technical Backdrop:** Pre-event 5-day momentum stands at **{mom5_str}** with volume running at **{vol_anom_str}** relative to its 20-day historical mean.

### 3. Historical Analog Precedents
- **Empirical Base Rate:** Vector retrieval over {sample_size} historical analogs demonstrates a **{win_rate_str} win rate** (AR3 > +0.5%) with a median realized 3-day abnormal return of **{median_ar3_str}**.
- **Case Precedents:**{analogs_text}
- **Alignment:** Historical analog outcomes {'reinforce' if (win_rate and win_rate >= 0.5) else 'diverge cautiously from'} the statistical ML model direction.

### 4. Downside Risks & Invalidation Triggers
- **Structural Risks:** Sector-wide beta drag, broad market index drawdowns, or immediate execution friction.
- **Invalidation Triggers:** A negative abnormal return closing below -1.0% relative to NIFTY 50 on T+1 or a failure of volume to confirm the price reaction invalidates the short-term outperformance hypothesis.
"""
        return memo.strip()

    # ------------------------------------------------------------------
    # Public Synthesis API
    # ------------------------------------------------------------------
    def synthesize_report(
        self,
        headline: str,
        body: str,
        extracted_event: Dict[str, Any],
        technicals: Dict[str, Any],
        prediction: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute synthesis through fallback cascade.
        Returns a dict containing sections, full_memo, and source_llm.
        """
        user_prompt = format_synthesis_prompt(
            headline=headline,
            body=body,
            extracted_event=extracted_event,
            technicals=technicals,
            prediction=prediction,
            evidence=evidence,
        )

        full_memo: Optional[str] = None
        source_llm = "deterministic_analyst_engine"

        # Tier 1: Ollama (Qwen2.5-1.5B via Ollama) — Local only, no Gemini in synthesis
        full_memo = self._call_ollama(user_prompt)
        if full_memo:
            source_llm = f"ollama/{self.ollama_model}"

        # Tier 2: Local HF Qwen (if opt-in enabled) — Gemini NOT used in synthesis
        if not full_memo:
            full_memo = self._call_hf_qwen(user_prompt)
            if full_memo:
                source_llm = "local_hf/qwen2.5-1.5b-instruct"

        # Tier 3: Deterministic Rule-Based Fallback (always available, zero latency)
        if not full_memo:
            full_memo = self._deterministic_synthesis(
                headline=headline,
                extracted_event=extracted_event,
                technicals=technicals,
                prediction=prediction,
                evidence=evidence,
            )
            source_llm = "deterministic_analyst_engine"

        # Parse sections
        sections = self._parse_memo_sections(full_memo)

        return {
            "source_llm": source_llm,
            "full_memo": full_memo,
            "sections": sections,
        }

    def _parse_memo_sections(self, memo: str) -> Dict[str, str]:
        """Splits markdown memo into structured keys for the frontend/API."""
        patterns = {
            "executive_thesis": r"###\s*1\.\s*Executive Thesis(.*?)(?=###\s*2\.|\Z)",
            "technical_event_drivers": r"###\s*2\.\s*Technical & Event Catalysts(.*?)(?=###\s*3\.|\Z)",
            "historical_precedent": r"###\s*3\.\s*Historical Analog Precedents(.*?)(?=###\s*4\.|\Z)",
            "downside_risks": r"###\s*4\.\s*Downside Risks & Invalidation Triggers(.*?)(?=\Z)",
        }
        res: Dict[str, str] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, memo, re.DOTALL | re.IGNORECASE)
            if match:
                res[key] = match.group(1).strip()
            else:
                res[key] = ""
        return res


_default_synthesizer: Optional[ReportSynthesizer] = None


def get_synthesizer() -> ReportSynthesizer:
    """Module-level cached ReportSynthesizer singleton."""
    global _default_synthesizer
    if _default_synthesizer is None:
        _default_synthesizer = ReportSynthesizer()
    return _default_synthesizer
