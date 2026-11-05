"""
backend/services/price_forecast_service.py — MarketLens Price Forecast Bridge
==============================================================================
Bridges the standalone 'stock-price-prediction-ml' ML project into the
MarketLens backend.

Features:
  - Loads pre-trained Random Forest / XGBoost / LSTM models from saved_models/
  - Fetches live market data from Yahoo Finance (with caching)
  - Runs the full feature engineering + forecast pipeline
  - Returns structured JSON-serialisable forecast dict
  - Graceful fallback if models are not found for the requested ticker

NOTE: Gemini is NOT used here. This is a fully local, offline-capable pipeline.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrap — add the stock-price-prediction-ml root to sys.path so
# that its internal imports (config, src.*) resolve correctly.
# ---------------------------------------------------------------------------
_ML_PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "ml_genai"
    / "stock-price-prediction-ml"
)

if str(_ML_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Model preference order — RF is fast + accurate; LSTM is deep but heavier
# ---------------------------------------------------------------------------
_PREFERRED_MODELS = ["Random Forest", "XGBoost", "LSTM"]


def _get_available_models(ticker: str) -> List[str]:
    """Return model names for which a saved artefact exists for *ticker*."""
    try:
        import config as ml_config  # stock-price-prediction-ml/config.py
        available = []
        for mname in _PREFERRED_MODELS:
            safe = mname.replace(" ", "_")
            if mname == "LSTM":
                p = ml_config.MODELS_DIR / f"{ticker.upper()}_{safe}_best.h5"
            else:
                p = ml_config.MODELS_DIR / f"{ticker.upper()}_{safe}.pkl"
            if p.exists():
                available.append(mname)
        return available
    except Exception as e:
        logger.debug("Could not check available models: %s", e)
        return []


def run_price_forecast(
    ticker: str,
    horizon_days: int = 7,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the stock-price-prediction-ml pipeline for *ticker* and return
    a structured forecast dict.

    Parameters
    ----------
    ticker:       NSE ticker without suffix (e.g. "RELIANCE", "TCS")
    horizon_days: Number of future trading days to forecast (default 7)
    model_name:   Which model to use. If None, picks best available.

    Returns
    -------
    dict with keys:
      ticker, model_used, last_actual_date, last_actual_close,
      forecast_dates (list[str]), forecast_prices (list[float]),
      signal ("Buy"/"Sell"/"Hold"),
      confidence_pct (int),
      price_change_pct (float),
      model_metrics (dict),
      error (str | None)
    """
    ticker = ticker.upper().strip()

    # -- Check available models --
    available = _get_available_models(ticker)
    if not available:
        return _error_response(
            ticker,
            f"No pre-trained models found for {ticker}. "
            f"Only RELIANCE models are pre-trained. "
            f"Please train models for {ticker} first.",
        )

    if model_name and model_name not in available:
        model_name = None  # fall back to auto-select

    if not model_name:
        model_name = available[0]  # prefer RF > XGBoost > LSTM

    logger.info("PriceForecast: ticker=%s model=%s horizon=%d", ticker, model_name, horizon_days)

    try:
        import config as ml_config
        from src.newpipeline import StockPredictionPipeline
        from src.forecasting import future_close_forecast, signal_for_forecast_window

        # Saved artefacts were trained with --sentiment (see TRAINING_MODELS /
        # start_training_if_needed above), which appends 5 extra feature columns
        # (sentiment_mean, sentiment_std, news_count, positive_ratio,
        # negative_ratio). Inference MUST request the same feature set or the
        # model raises "X has N features, but ... expecting M features".
        pipeline = StockPredictionPipeline(
            ticker=ticker,
            model_names=[model_name],
            use_sentiment=True,
            save_models=False,
        )
        evaluator = pipeline.run()

        # Get metrics for this model
        metrics_dict: Dict[str, Any] = {}
        try:
            tbl = evaluator.comparison_table()
            row = tbl[tbl["Model"] == model_name]
            if not row.empty:
                metrics_dict = {
                    "MAE":  round(float(row["MAE"].iloc[0]), 2),
                    "RMSE": round(float(row["RMSE"].iloc[0]), 2),
                    "MAPE": round(float(row["MAPE"].iloc[0]), 2),
                    "R2":   round(float(row["R2"].iloc[0]), 4),
                }
        except Exception:
            pass

        # Get clean_df for forecasting
        from src.data_collection import StockDataFetcher
        from src.preprocessing import DataPreprocessor
        raw_df = StockDataFetcher(ticker).fetch()
        preprocessor = DataPreprocessor()
        clean_df = preprocessor.clean(raw_df)

        # Re-fit scaler so we can forecast — build with the same zero-score
        # sentiment columns the saved model was trained on (see note above).
        from src.feature_engineering import FeatureEngineer
        from src.sentiment import SentimentAnalyzer
        engineer = FeatureEngineer()
        sentiment_df = SentimentAnalyzer.make_zero_scores(clean_df.index)
        feature_df = engineer.build_features(clean_df, sentiment_df)
        feature_cols = [c for c in feature_df.columns if c != "Close"]
        all_cols = feature_cols + ["Close"]
        preprocessor.fit_scale(feature_df, all_cols)

        is_lstm = model_name == "LSTM"
        test_r2 = metrics_dict.get("R2", 0.0)

        lstm_ret_scaler = None
        if is_lstm and ml_config.LSTM_TARGET_MODE == "return":
            lstm_ret_scaler = pipeline.lstm_ret_scaler

        trained_model = pipeline.trained_models[model_name]

        n_steps = min(horizon_days, ml_config.FORECAST_MAX_HORIZON)
        fc = future_close_forecast(
            clean_df=clean_df,
            engineer=engineer,
            preprocessor=preprocessor,
            sentiment_df=sentiment_df,
            model=trained_model,
            is_lstm=is_lstm,
            feature_cols=feature_cols,
            n_steps=n_steps,
            look_back=ml_config.LOOK_BACK,
            test_r2=test_r2,
            lstm_univariate=(is_lstm and ml_config.LSTM_UNIVARIATE_INPUT),
            lstm_target_return=(is_lstm and ml_config.LSTM_TARGET_MODE == "return"),
            lstm_ret_scaler=lstm_ret_scaler,
            tabular_target_return=(not is_lstm and ml_config.TABULAR_TARGET_MODE == "return"),
        )

        # Serialise dates to ISO strings
        forecast_dates = [d.strftime("%Y-%m-%d") for d in fc["dates"]]
        forecast_prices = [round(float(p), 2) for p in fc["prices"]]
        last_close = float(fc["last_actual_close"])
        last_date = fc["last_actual_date"].strftime("%Y-%m-%d")

        # Price change over forecast window
        first_pred = forecast_prices[0] if forecast_prices else last_close
        last_pred = forecast_prices[-1] if forecast_prices else last_close
        price_change_pct = round((last_pred - last_close) / last_close * 100, 2)

        return {
            "ticker": ticker,
            "model_used": model_name,
            "last_actual_date": last_date,
            "last_actual_close": round(last_close, 2),
            "forecast_dates": forecast_dates,
            "forecast_prices": forecast_prices,
            "signal": fc["signal"],
            "confidence_pct": fc["confidence_pct"],
            "price_change_pct": price_change_pct,
            "model_metrics": metrics_dict,
            "available_models": available,
            "error": None,
        }

    except Exception as e:
        logger.exception("PriceForecast pipeline error for %s: %s", ticker, e)
        return _error_response(ticker, str(e))


# ---------------------------------------------------------------------------
# On-demand background training — triggered when a chat/forecast request
# names a ticker with no pre-trained artefacts yet.
# ---------------------------------------------------------------------------
TRAINING_START_DATE = "2021-01-01"
TRAINING_MODELS = ["Random Forest", "XGBoost", "LSTM"]

_training_lock = threading.Lock()
_training_status: Dict[str, Dict[str, Any]] = {}  # ticker -> {status, started_at, error}


def get_training_status(ticker: str) -> Optional[Dict[str, Any]]:
    """Return the current background-training status dict for *ticker*, or None."""
    return _training_status.get(ticker.upper().strip())


def start_training_if_needed(ticker: str) -> bool:
    """
    Kick off background training for *ticker* (Random Forest + XGBoost + LSTM,
    2021-01-01 to today, with sentiment features) if no job is already running.

    Mirrors: python -m src.newpipeline --ticker TICKER --start 2021-01-01
             --end <today> --models "Random Forest" XGBoost LSTM --sentiment

    Returns True if a new job was started, False if one was already in flight.
    """
    ticker = ticker.upper().strip()
    with _training_lock:
        existing = _training_status.get(ticker)
        if existing and existing.get("status") == "training":
            return False
        _training_status[ticker] = {"status": "training", "started_at": time.time(), "error": None}

    def _run() -> None:
        try:
            logger.info("Background training started for %s (%s)", ticker, TRAINING_MODELS)
            from src.newpipeline import StockPredictionPipeline

            pipeline = StockPredictionPipeline(
                ticker=ticker,
                start_date=TRAINING_START_DATE,
                end_date=date.today().isoformat(),
                model_names=TRAINING_MODELS,
                use_sentiment=True,
                save_models=True,
            )
            pipeline.run()
            with _training_lock:
                started_at = _training_status.get(ticker, {}).get("started_at", time.time())
                _training_status[ticker] = {"status": "done", "started_at": started_at, "error": None}
            logger.info("Background training complete for %s", ticker)
        except Exception as e:
            logger.exception("Background training failed for %s: %s", ticker, e)
            with _training_lock:
                started_at = _training_status.get(ticker, {}).get("started_at", time.time())
                _training_status[ticker] = {"status": "failed", "started_at": started_at, "error": str(e)}

    threading.Thread(target=_run, daemon=True, name=f"train-{ticker}").start()
    return True


def _error_response(ticker: str, error_msg: str) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "model_used": None,
        "last_actual_date": None,
        "last_actual_close": None,
        "forecast_dates": [],
        "forecast_prices": [],
        "signal": "Hold",
        "confidence_pct": 0,
        "price_change_pct": 0.0,
        "model_metrics": {},
        "available_models": _get_available_models(ticker),
        "error": error_msg,
    }
