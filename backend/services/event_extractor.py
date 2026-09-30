"""
Phase 2 Hybrid NLP Event Extractor Service — 100% Local / Offline

Extraction engine:
  - Qwen/Qwen2.5-1.5B-Instruct (local HuggingFace Transformers)
    Structured JSON extraction via chat-template + constrained prompt.
    No API key, no cloud dependency, runs fully offline.

Sentiment scoring:
  - ProsusAI/FinBERT (local HuggingFace Transformers)
    Continuous float probabilities: positive, negative, neutral.

Both models are loaded once via lazy singletons to avoid reloading on
every article. All sessions are caller-managed; this service is stateless.
"""
import json
import logging
import re
import time
from typing import Optional

from sqlalchemy.orm import Session

from backend.services.event_schemas import FinancialEventExtraction

logger = logging.getLogger(__name__)

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Qwen2.5 configuration
# ---------------------------------------------------------------------------
def _resolve_qwen_path() -> str:
    """Return local path if downloaded, else env var or HuggingFace hub id."""
    env_path = os.getenv("QWEN_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Check repository models/ directory
    repo_model_dir = Path(__file__).resolve().parent.parent.parent / "models" / "Qwen2.5-1.5B-Instruct"
    if repo_model_dir.is_dir() and (repo_model_dir / "config.json").is_file():
        return str(repo_model_dir)

    return env_path or "Qwen/Qwen2.5-1.5B-Instruct"

QWEN_MODEL_NAME     = _resolve_qwen_path()
QWEN_MAX_NEW_TOKENS = 400     # Enough for the full JSON structure
QWEN_TEMPERATURE    = 0.1     # Near-deterministic for structured extraction

# Lazily initialised singletons
_qwen_tokenizer = None
_qwen_model     = None
_qwen_device    = "cpu"    # Set during _load_qwen(); used in _call_qwen()

# ---------------------------------------------------------------------------
# FinBERT configuration
# ---------------------------------------------------------------------------
FINBERT_MODEL_NAME  = "ProsusAI/finbert"
_FINBERT_MAX_TOKENS = 512

_finbert_tokenizer = None
_finbert_model     = None

# ---------------------------------------------------------------------------
# Valid values for Pydantic Literal fields (used in prompt + fallback)
# ---------------------------------------------------------------------------
_VALID_EVENT_TYPES = {
    "EARNINGS", "MONETARY_POLICY", "REGULATION", "CONTRACT_ORDER",
    "MERGER_ACQUISITION", "MANAGEMENT_CHANGE", "MACRO_DATA", "OTHER",
}
_VALID_SENTIMENTS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}
_VALID_SURPRISES  = {"BEAT", "MISS", "IN_LINE", "NOT_APPLICABLE"}

# ---------------------------------------------------------------------------
# System + user prompt templates
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a senior financial analyst specialising in Indian equity markets (NSE/BSE).
Analyse the provided news headline (and article body if available) and extract
structured financial intelligence as a single JSON object.

Strict rules:
1. "ticker": NSE ticker WITHOUT suffix (e.g. "INFY", "TCS"). null if macro/RBI/SEBI news.
2. "event_type": one of EARNINGS | MONETARY_POLICY | REGULATION | CONTRACT_ORDER |
   MERGER_ACQUISITION | MANAGEMENT_CHANGE | MACRO_DATA | OTHER
3. "sentiment": POSITIVE | NEGATIVE | NEUTRAL (likely near-term market impact)
4. "magnitude": numeric value only (e.g. 50 for 50 bps, 15 for 15% growth). null if N/A.
5. "surprise": BEAT | MISS | IN_LINE | NOT_APPLICABLE
6. "summary": ONE sentence, max 120 chars, stating the core financial fact.

Reply with ONLY the raw JSON object. No markdown, no explanation, no extra text."""

_USER_TEMPLATE = """\
Headline: {headline}
{body_section}
Extract the financial event as JSON:"""


# ---------------------------------------------------------------------------
# Qwen2.5 helpers
# ---------------------------------------------------------------------------

def _load_qwen() -> None:
    """Lazy-load Qwen2.5-1.5B-Instruct. Runs once per process."""
    global _qwen_tokenizer, _qwen_model, _qwen_device
    if _qwen_tokenizer is not None:
        return

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    _qwen_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Loading Qwen model '%s' on device=%s (one-time startup)...",
        QWEN_MODEL_NAME, _qwen_device,
    )

    is_local = os.path.exists(QWEN_MODEL_NAME)
    load_kwargs = {"trust_remote_code": True}
    if is_local:
        load_kwargs["local_files_only"] = True

    # Disable sliding_window to avoid slow CPU SDPA fallback loop
    cfg = AutoConfig.from_pretrained(QWEN_MODEL_NAME, **load_kwargs)
    cfg.sliding_window = None
    cfg.use_sliding_window = False

    _qwen_tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_NAME, **load_kwargs
    )
    # Use bfloat16 on CPU for ~7x speedup and lower memory footprint, float16 on CUDA
    target_dtype = torch.float16 if _qwen_device == "cuda" else torch.bfloat16

    _qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        config=cfg,
        torch_dtype=target_dtype,
        **load_kwargs,
    )
    _qwen_model = _qwen_model.to(_qwen_device)
    _qwen_model.eval()
    logger.info("Qwen model ready on device=%s (%s).", _qwen_device, target_dtype)


def _extract_json_from_text(text: str) -> dict:
    """
    Robustly extract the first valid JSON object from model output.

    Tries three strategies in order:
      1. Direct json.loads (model output is clean JSON)
      2. Regex extraction between first '{' and last '}'
      3. Regex for a fenced ```json ... ``` block
    """
    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract between outermost braces
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3: fenced code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in model output:\n{text[:500]}")


def _sanitise_extraction(data: dict) -> dict:
    """
    Normalise and clamp model output values to valid Pydantic Literal ranges
    before validation, preventing hard ValidationErrors from minor model mistakes.
    """
    # Uppercase all enum fields
    for field in ("event_type", "sentiment", "surprise"):
        if isinstance(data.get(field), str):
            data[field] = data[field].strip().upper().replace(" ", "_")

    # Clamp to valid sets
    if data.get("event_type") not in _VALID_EVENT_TYPES:
        data["event_type"] = "OTHER"
    if data.get("sentiment") not in _VALID_SENTIMENTS:
        data["sentiment"] = "NEUTRAL"
    if data.get("surprise") not in _VALID_SURPRISES:
        data["surprise"] = "NOT_APPLICABLE"

    # Normalise ticker: strip suffix, uppercase, null if empty string
    ticker = data.get("ticker")
    if ticker:
        ticker = ticker.split(".")[0].strip().upper()
        data["ticker"] = ticker if ticker else None

    # Normalise magnitude: must be numeric or null
    magnitude = data.get("magnitude")
    if magnitude is not None:
        try:
            data["magnitude"] = float(magnitude)
        except (TypeError, ValueError):
            data["magnitude"] = None

    # Ensure summary is a non-empty string
    if not data.get("summary"):
        data["summary"] = "Financial event detected."

    return data


def _call_qwen(headline: str, article_body: Optional[str], max_retries: int = 2) -> FinancialEventExtraction:
    """
    Run Qwen2.5-1.5B-Instruct inference to extract a FinancialEventExtraction.

    On JSON parse failure, retries up to max_retries times with a brief pause.

    Args:
        headline:     News headline text.
        article_body: Optional article body (first 1,500 chars used).
        max_retries:  Retry count on parse failure (JSON/validation errors).

    Returns:
        Validated FinancialEventExtraction Pydantic model.

    Raises:
        ValueError: If all retries are exhausted without a valid extraction.
    """
    import torch

    _load_qwen()

    body_section = ""
    if article_body and article_body.strip():
        body_section = f"Article body (excerpt):\n{article_body.strip()[:1500]}"

    user_content = _USER_TEMPLATE.format(
        headline=headline,
        body_section=body_section,
    )

    # Build chat messages using Qwen2.5-Instruct chat template
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            # Tokenize using the model's built-in chat template
            inputs = _qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = inputs["input_ids"].to(_qwen_device)
            attention_mask = inputs["attention_mask"].to(_qwen_device)

            with torch.no_grad():
                output_ids = _qwen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=150,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            # Decode only the newly generated tokens (skip the prompt)
            new_tokens = output_ids[0][input_ids.shape[-1]:]
            raw_output = _qwen_tokenizer.decode(new_tokens, skip_special_tokens=True)

            logger.debug("[Qwen] Raw output (attempt %d): %s", attempt + 1, raw_output[:300])

            # Parse and validate
            raw_dict  = _extract_json_from_text(raw_output)
            clean_dict = _sanitise_extraction(raw_dict)
            return FinancialEventExtraction.model_validate(clean_dict)

        except Exception as exc:
            last_error = exc
            logger.warning(
                "[Qwen] Parse/validation failed (attempt %d/%d): %s",
                attempt + 1, max_retries + 1, exc,
            )
            if attempt < max_retries:
                time.sleep(1)   # brief pause before retry

    raise ValueError(
        f"Qwen extraction failed after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# FinBERT helpers
# ---------------------------------------------------------------------------

def _load_finbert() -> None:
    """Lazy-load FinBERT tokenizer and model. Runs once per process."""
    global _finbert_tokenizer, _finbert_model
    if _finbert_tokenizer is not None:
        return

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading FinBERT model '%s' (one-time startup)...", FINBERT_MODEL_NAME)
    _finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    _finbert_model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    _finbert_model.eval()
    logger.info("FinBERT loaded successfully.")


def get_finbert_scores(text: str) -> dict[str, float]:
    """
    Run FinBERT on *text* and return softmax probabilities.

    FinBERT label order: positive=0, negative=1, neutral=2.

    Returns:
        Dict with keys 'positive', 'negative', 'neutral' summing to ~1.0.
    """
    import torch

    _load_finbert()

    inputs = _finbert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=_FINBERT_MAX_TOKENS,
        padding=True,
    )

    with torch.no_grad():
        outputs = _finbert_model(**inputs)
        probs   = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()

    return {
        "positive": float(probs[0]),
        "negative": float(probs[1]),
        "neutral":  float(probs[2]),
    }


# ---------------------------------------------------------------------------
# Active ticker cache
# ---------------------------------------------------------------------------

def _fetch_active_tickers(session: Session) -> set[str]:
    """Return the set of active ticker symbols from the companies table."""
    from database.models import Company
    rows = session.query(Company.ticker).filter(Company.is_active.is_(True)).all()
    return {row.ticker for row in rows}


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class EventExtractorService:
    """
    Orchestrates local Qwen2.5 structured extraction + FinBERT sentiment scoring.

    Both models are loaded lazily on first call and reused for the lifetime
    of the service instance (instantiate once per script run).

    Usage::

        service = EventExtractorService()
        result  = service.extract(
            news_id=42,
            headline="Infosys Q2 profit rises 5%, beats estimates",
            article_body="...",
            session=db_session,
        )
    """

    def __init__(self) -> None:
        logger.info(
            "EventExtractorService ready — extraction: %s | sentiment: %s",
            QWEN_MODEL_NAME, FINBERT_MODEL_NAME,
        )

    # ------------------------------------------------------------------
    # Ticker FK safety
    # ------------------------------------------------------------------

    def _validate_ticker(
        self,
        ticker: Optional[str],
        active_tickers: set[str],
    ) -> Optional[str]:
        """
        Validate the extracted ticker against the live companies table.
        Strips any exchange suffix before checking.

        Returns the validated base ticker, or None if not found in DB.
        """
        if not ticker:
            return None

        base = ticker.split(".")[0].strip().upper()
        if base in active_tickers:
            return base

        logger.warning(
            "Extracted ticker=%r (normalised=%r) not in active companies — "
            "setting to None to prevent FK violation.",
            ticker, base,
        )
        return None

    # ------------------------------------------------------------------
    # Public extraction entry point
    # ------------------------------------------------------------------

    def extract(
        self,
        news_id: int,
        headline: str,
        article_body: Optional[str],
        session: Session,
    ) -> dict:
        """
        Full extraction pipeline for a single news article.

        Steps:
          1. Qwen2.5-1.5B-Instruct → FinancialEventExtraction (JSON → Pydantic)
          2. Ticker validation against active companies DB
          3. FinBERT → continuous sentiment probabilities
          4. Return assembled dict ready for Event(**dict) construction

        Args:
            news_id:      PK of the source News row.
            headline:     Article headline.
            article_body: Full article body, or None.
            session:      Active SQLAlchemy session (read-only here).

        Returns:
            Dict of Event field values.

        Raises:
            ValueError: If Qwen extraction fails after all retries.
        """
        # 1. Local Qwen extraction
        logger.debug("[news_id=%d] Running Qwen extraction...", news_id)
        extraction = _call_qwen(headline, article_body)
        logger.debug("[news_id=%d] Extraction: %s", news_id, extraction.model_dump())

        # 2. FK-safe ticker resolution
        active_tickers = _fetch_active_tickers(session)
        safe_ticker    = self._validate_ticker(extraction.ticker, active_tickers)

        # 3. FinBERT scoring
        finbert_input = headline
        if article_body and article_body.strip():
            finbert_input = f"{headline}. {article_body.strip()[:400]}"

        logger.debug("[news_id=%d] Running FinBERT...", news_id)
        finbert_scores = get_finbert_scores(finbert_input)

        # 4. Assemble result dict
        return {
            "news_id":          news_id,
            "ticker":           safe_ticker,
            "event_type":       extraction.event_type,
            "gemini_sentiment": extraction.sentiment,   # column name kept for schema compat
            "magnitude":        extraction.magnitude,
            "surprise":         extraction.surprise if extraction.surprise != "NOT_APPLICABLE" else None,
            "summary":          extraction.summary,
            "finbert_positive": finbert_scores["positive"],
            "finbert_negative": finbert_scores["negative"],
            "finbert_neutral":  finbert_scores["neutral"],
        }
