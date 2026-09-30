"""
backend/ml/features.py — MarketLens Phase 3
=============================================
Feature engineering for the financial ML pipeline.

All features are computed using ONLY data available up to P0 (benchmark price date).
This guarantees zero look-ahead bias for downstream model training.

Technical/Market Features:
  momentum_5d     : 5-day rolling stock return up to P0
  momentum_20d    : 20-day rolling stock return up to P0
  volatility_20d  : 20-day rolling std-dev of daily stock returns up to P0
  volume_anomaly  : Recent volume / 20-day SMA of volume
  market_return_5d: 5-day rolling return of NIFTY50 up to m0 date

Event Features (from Phase 2 events table):
  finbert_positive, finbert_negative, finbert_neutral  (continuous)
  magnitude       : numeric (missing -> 0.0)
  event_type      : one-hot encoded
  surprise        : one-hot encoded
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid categoricals (must match Phase 2 schema)
# ---------------------------------------------------------------------------
EVENT_TYPES = [
    "EARNINGS", "MONETARY_POLICY", "REGULATION", "CONTRACT_ORDER",
    "MERGER_ACQUISITION", "MANAGEMENT_CHANGE", "MACRO_DATA", "OTHER",
]
SURPRISE_TYPES = ["BEAT", "MISS", "IN_LINE", "NOT_APPLICABLE"]


# ---------------------------------------------------------------------------
# Technical feature computation (per-ticker time series)
# ---------------------------------------------------------------------------

def compute_technical_features(
    mkt_df: pd.DataFrame,
    idx_df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join technical/market features onto the event matrix.

    For each event row, all lookback windows use only data strictly on or
    before the P0 benchmark date (event_date - 1 trading day).

    Parameters
    ----------
    mkt_df : DataFrame with columns [ticker, date, close, volume]
    idx_df : DataFrame with columns [date, index_close]
    event_df : Event matrix from dataset_builder.build_event_matrix()

    Returns
    -------
    event_df with additional columns:
        momentum_5d, momentum_20d, volatility_20d,
        volume_anomaly, market_return_5d
    """
    # Pre-compute per-ticker sorted history (avoid repeated filtering)
    ticker_histories: dict[str, pd.DataFrame] = {}
    for ticker in mkt_df["ticker"].unique():
        t_df = mkt_df[mkt_df["ticker"] == ticker].sort_values("date").copy()
        t_df["ret"] = t_df["close"].pct_change()
        ticker_histories[ticker] = t_df

    # Pre-compute index returns
    idx_df = idx_df.sort_values("date").copy()
    idx_df["idx_ret"] = idx_df["index_close"].pct_change()

    results = []
    for _, row in event_df.iterrows():
        ticker = row["ticker"]
        # P0 date is the day before event_date in trading calendar
        # event_date = T0 (reaction day); P0 date = T0 - 1 trading day
        # The mkt_df already has the actual trading dates
        p0_cutoff = row["event_date"]  # inclusive cutoff: use data UP TO but NOT INCLUDING T0

        if ticker not in ticker_histories:
            results.append(_nan_technical_row())
            continue

        hist = ticker_histories[ticker]
        # Strictly use data up to (but not including) the reaction day T0
        past = hist[hist["date"] < p0_cutoff]

        if len(past) < 5:
            results.append(_nan_technical_row())
            continue

        close_series = past["close"].values
        ret_series = past["ret"].values
        vol_series = past["volume"].values

        # --- Momentum ---
        mom_5d  = _rolling_return(close_series, 5)
        mom_20d = _rolling_return(close_series, 20)

        # --- Volatility: 20-day std of daily returns ---
        vol_20d = _rolling_std(ret_series, 20)

        # --- Volume anomaly: recent vol / 20-day SMA ---
        recent_vol = float(vol_series[-1]) if len(vol_series) > 0 else np.nan
        sma_vol = np.mean(vol_series[-20:]) if len(vol_series) >= 5 else np.nan
        vol_anomaly = recent_vol / sma_vol if sma_vol and sma_vol > 0 else np.nan

        # --- Market return: 5-day NIFTY50 return up to P0 date ---
        idx_past = idx_df[idx_df["date"] < p0_cutoff]
        idx_ret_5d = _rolling_return(idx_past["index_close"].values, 5)

        results.append({
            "momentum_5d":       mom_5d,
            "momentum_20d":      mom_20d,
            "volatility_20d":    vol_20d,
            "volume_anomaly":    vol_anomaly,
            "market_return_5d":  idx_ret_5d,
        })

    tech_df = pd.DataFrame(results, index=event_df.index)
    return pd.concat([event_df, tech_df], axis=1)


def _rolling_return(prices: np.ndarray, window: int) -> float:
    """Return = (last / last-window) - 1. NaN if insufficient data."""
    if len(prices) <= window:
        return np.nan
    return float(prices[-1] / prices[-window - 1] - 1) if prices[-window - 1] != 0 else np.nan


def _rolling_std(returns: np.ndarray, window: int) -> float:
    """Rolling std of last `window` returns. NaN if insufficient data."""
    if len(returns) < window:
        return np.nan
    return float(np.std(returns[-window:], ddof=1))


def _nan_technical_row() -> dict:
    return {
        "momentum_5d":      np.nan,
        "momentum_20d":     np.nan,
        "volatility_20d":   np.nan,
        "volume_anomaly":   np.nan,
        "market_return_5d": np.nan,
    }


# ---------------------------------------------------------------------------
# Event feature preparation (encoding & imputation)
# ---------------------------------------------------------------------------

def prepare_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical event features and impute numeric features.

    Actions:
      - magnitude: fillna(0.0)
      - surprise:  fillna("NOT_APPLICABLE"); one-hot encode
      - event_type: one-hot encode
      - finbert columns: pass through (already float)

    Returns the enriched DataFrame.
    """
    df = df.copy()

    # Impute magnitude
    df["magnitude"] = df["magnitude"].fillna(0.0)

    # Normalise surprise (some may be None/NaN)
    df["surprise"] = df["surprise"].fillna("NOT_APPLICABLE").str.upper()

    # One-hot: event_type
    for et in EVENT_TYPES:
        df[f"evt_{et.lower()}"] = (df["event_type"] == et).astype(int)

    # One-hot: surprise
    for sv in SURPRISE_TYPES:
        df[f"surp_{sv.lower()}"] = (df["surprise"] == sv).astype(int)

    return df


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_features(
    event_df: pd.DataFrame,
    mkt_df: pd.DataFrame,
    idx_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
      1. Compute technical features (momentum, volatility, volume, market return)
      2. Encode event features (one-hot, imputation)
      3. Drop rows with NaN in critical technical features
      4. Return clean feature-enriched DataFrame

    Parameters
    ----------
    event_df : Raw event matrix from dataset_builder.build_event_matrix()
    mkt_df   : Market data DataFrame [ticker, date, close, volume]
    idx_df   : Index data DataFrame [date, index_close]

    Returns
    -------
    DataFrame with full feature set; rows with insufficient history dropped.
    """
    logger.info("Computing technical features for %d events...", len(event_df))
    df = compute_technical_features(mkt_df, idx_df, event_df)

    logger.info("Encoding event features...")
    df = prepare_event_features(df)

    # Drop rows missing critical technical lookbacks
    critical = ["momentum_5d", "volatility_20d"]
    before = len(df)
    df = df.dropna(subset=critical).reset_index(drop=True)
    after = len(df)
    logger.info(
        "Dropped %d rows with NaN in critical features (%d -> %d rows).",
        before - after, before, after,
    )

    return df


# ---------------------------------------------------------------------------
# Feature column definitions (for train.py)
# ---------------------------------------------------------------------------

TECHNICAL_FEATURES = [
    "momentum_5d",
    "momentum_20d",
    "volatility_20d",
    "volume_anomaly",
    "market_return_5d",
]

SENTIMENT_FEATURES = [
    "finbert_positive",
    "finbert_negative",
    "finbert_neutral",
]

EVENT_TYPE_FEATURES = [f"evt_{et.lower()}" for et in EVENT_TYPES]
SURPRISE_FEATURES   = [f"surp_{sv.lower()}" for sv in SURPRISE_TYPES]

EVENT_NUMERIC_FEATURES = ["magnitude"]

# Feature sets for the 3-model ablation experiment
FEATURE_SET_A = TECHNICAL_FEATURES                                    # Market Baseline
FEATURE_SET_B = TECHNICAL_FEATURES + SENTIMENT_FEATURES               # + FinBERT
FEATURE_SET_C = (                                                      # + Event-Aware
    TECHNICAL_FEATURES
    + SENTIMENT_FEATURES
    + EVENT_TYPE_FEATURES
    + SURPRISE_FEATURES
    + EVENT_NUMERIC_FEATURES
)

TARGET_COL = "target_AR3_positive"


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from backend.ml.dataset_builder import (
        build_event_matrix,
        _load_market_data,
        _load_index_data,
    )
    from database.connection import SessionLocal

    with SessionLocal() as session:
        event_df = build_event_matrix(session)
        mkt_df   = _load_market_data(session)
        idx_df   = _load_index_data(session)

    print(f"Raw event matrix: {event_df.shape}")
    df = build_features(event_df, mkt_df, idx_df)
    print(f"After feature engineering: {df.shape}")
    print("\nFeature summary:")
    print(df[FEATURE_SET_C].describe().to_string())
    print(f"\nTarget distribution:\n{df[TARGET_COL].value_counts()}")
    print(f"\nFeature Set A ({len(FEATURE_SET_A)} features): {FEATURE_SET_A}")
    print(f"Feature Set B ({len(FEATURE_SET_B)} features): {FEATURE_SET_B}")
    print(f"Feature Set C ({len(FEATURE_SET_C)} features):")
    for f in FEATURE_SET_C:
        print(f"  - {f}")
