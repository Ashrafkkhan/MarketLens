"""
scripts/test_prediction.py — MarketLens Live Inference Tester
=============================================================
Standalone, interactive verification script for Model C pipeline
(models/event_classifier_v1.joblib).

Evaluates whether financial event intelligence + market price features
predict a 3-day abnormal return outperformance (AR3 > +0.5% vs NIFTY 50).

Usage:
    python scripts/test_prediction.py                  # Run 3 built-in scenarios
    python scripts/test_prediction.py --interactive    # Prompt for custom event features
    python scripts/test_prediction.py --threshold 0.60 # Custom decision boundary
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Suppress unpickling version warnings across sklearn versions
warnings.filterwarnings("ignore")

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants & Taxonomy
# ---------------------------------------------------------------------------
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "models" / "event_classifier_v1.joblib"
TARGET_NAME = "3-Day Abnormal Return Outperformance vs NIFTY 50 (> +0.5%)"

EVENT_TYPES = [
    "EARNINGS", "MONETARY_POLICY", "REGULATION", "CONTRACT_ORDER",
    "MERGER_ACQUISITION", "MANAGEMENT_CHANGE", "MACRO_DATA", "OTHER",
]
SURPRISE_TYPES = ["BEAT", "MISS", "IN_LINE", "NOT_APPLICABLE"]


# ---------------------------------------------------------------------------
# Artifact Loader & Validator
# ---------------------------------------------------------------------------

def load_and_validate_artifact(
    model_path: Path,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    Load and introspect the serialized pipeline artifact.
    Handles both raw Pipeline instances and metadata dictionary wrappers.

    Returns:
        (pipeline, feature_cols, metadata)
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Artifact not found at '{model_path}'.\n"
            f"Please run the training pipeline first: python backend/ml/train.py"
        )

    raw_artifact = joblib.load(model_path)
    metadata: Dict[str, Any] = {}
    pipeline = None
    feature_cols: List[str] = []

    if isinstance(raw_artifact, dict):
        pipeline = raw_artifact.get("pipeline") or raw_artifact.get("model")
        feature_cols = (
            raw_artifact.get("feature_cols")
            or raw_artifact.get("features")
            or raw_artifact.get("columns")
            or []
        )
        metadata = {
            k: v for k, v in raw_artifact.items() if k not in ["pipeline", "model"]
        }
    else:
        pipeline = raw_artifact

    if pipeline is None:
        raise ValueError(
            f"Could not extract a valid Scikit-Learn/XGBoost model or pipeline from {model_path}."
        )

    # Introspect pipeline if feature list was not explicitly saved in dictionary
    if not feature_cols:
        if hasattr(pipeline, "feature_names_in_"):
            feature_cols = list(pipeline.feature_names_in_)
        elif hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
            preprocessor = pipeline.named_steps["preprocessor"]
            if hasattr(preprocessor, "transformers_"):
                # ColumnTransformer (num, Pipeline, [cols])
                for name, trans, cols in preprocessor.transformers_:
                    if isinstance(cols, (list, tuple)):
                        feature_cols.extend(cols)

    if not feature_cols:
        raise ValueError(
            "Could not determine expected feature columns from artifact. "
            "Please ensure the artifact includes 'feature_cols' or uses Scikit-Learn feature_names_in_."
        )

    return pipeline, feature_cols, metadata


# ---------------------------------------------------------------------------
# Feature Matrix Builder with Validation & Diff Reporting
# ---------------------------------------------------------------------------

def build_scenario_dataframe(
    raw_inputs: Dict[str, Any],
    expected_cols: List[str],
) -> pd.DataFrame:
    """
    Converts raw feature dict into a single-row DataFrame strictly aligned
    with the pipeline's expected columns and schema.

    Catches missing/misaligned columns and provides clean error reporting.
    """
    row_data: Dict[str, float] = {}

    # 1. Base technical / market features
    row_data["momentum_5d"] = float(raw_inputs.get("momentum_5d", 0.0))
    row_data["momentum_20d"] = float(raw_inputs.get("momentum_20d", 0.0))
    row_data["volatility_20d"] = float(raw_inputs.get("volatility_20d", 0.015))
    row_data["volume_anomaly"] = float(raw_inputs.get("volume_anomaly", 1.0))
    row_data["market_return_5d"] = float(raw_inputs.get("market_return_5d", 0.0))

    # 2. Continuous sentiment
    row_data["finbert_positive"] = float(raw_inputs.get("finbert_positive", 0.0))
    row_data["finbert_negative"] = float(raw_inputs.get("finbert_negative", 0.0))
    row_data["finbert_neutral"] = float(raw_inputs.get("finbert_neutral", 0.0))

    # 3. Categoricals - Event Type one-hot
    event_type = str(raw_inputs.get("event_type", "OTHER")).upper()
    for et in EVENT_TYPES:
        key = f"evt_{et.lower()}"
        # Direct override from raw inputs if passed as evt_* or derived from event_type string
        if key in raw_inputs:
            row_data[key] = float(raw_inputs[key])
        else:
            row_data[key] = 1.0 if event_type == et else 0.0

    # 4. Categoricals - Surprise one-hot
    surprise = str(raw_inputs.get("surprise", "NOT_APPLICABLE")).upper()
    for sv in SURPRISE_TYPES:
        key = f"surp_{sv.lower()}"
        if key in raw_inputs:
            row_data[key] = float(raw_inputs[key])
        else:
            row_data[key] = 1.0 if surprise == sv else 0.0

    # 5. Event Numeric
    row_data["magnitude"] = float(raw_inputs.get("magnitude", 0.0))

    # Verify against expected pipeline columns
    df = pd.DataFrame([row_data])

    missing_cols = set(expected_cols) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_cols)

    if missing_cols:
        raise KeyError(
            f"Schema mismatch! Input DataFrame is missing required features: {sorted(list(missing_cols))}\n"
            f"Provided columns: {sorted(list(df.columns))}"
        )

    # Reorder strictly to pipeline expectations
    return df[expected_cols]


# ---------------------------------------------------------------------------
# Prediction & Structured Reporting
# ---------------------------------------------------------------------------

def run_prediction(
    pipeline: Any,
    feature_df: pd.DataFrame,
    threshold: float = 0.50,
) -> Tuple[int, float]:
    """
    Executes prediction and probability inference.
    Returns (predicted_class, prob_positive).
    """
    # Predict probabilities if supported
    if hasattr(pipeline, "predict_proba"):
        prob_matrix = pipeline.predict_proba(feature_df)
        prob_pos = float(prob_matrix[0, 1])
    else:
        # Fallback if model only outputs decision function or raw class
        pred = pipeline.predict(feature_df)[0]
        prob_pos = 1.0 if pred == 1 else 0.0

    # Apply decision threshold
    pred_class = 1 if prob_pos >= threshold else 0
    return pred_class, prob_pos


def print_prediction_card(
    scenario_title: str,
    raw_inputs: Dict[str, Any],
    pred_class: int,
    prob_pos: float,
    threshold: float = 0.50,
) -> None:
    """
    Renders an institutional quantitative report card for a single inference.
    """
    ticker = raw_inputs.get("ticker", "N/A")
    event_type = raw_inputs.get("event_type", "N/A")
    surprise = raw_inputs.get("surprise", "N/A")
    pos_sent = raw_inputs.get("finbert_positive", 0.0)
    neg_sent = raw_inputs.get("finbert_negative", 0.0)
    vol_anom = raw_inputs.get("volume_anomaly", 1.0)
    mom_5d = raw_inputs.get("momentum_5d", 0.0)

    # Decision banner
    if pred_class == 1:
        decision_tag = "OUTPERFORM (+1)"
        decision_desc = "Probable positive alpha against NIFTY 50 (> +0.5% over 3 days)"
    else:
        decision_tag = "UNDERPERFORM / NEUTRAL (0)"
        decision_desc = "No statistically significant abnormal excess return anticipated"

    # Contextual risk commentary
    commentary = []
    if pos_sent >= 0.70:
        commentary.append(f"Strong positive FinBERT sentiment ({pos_sent*100:.1f}%) provides upside tailwind.")
    elif neg_sent >= 0.70:
        commentary.append(f"Severe negative sentiment ({neg_sent*100:.1f}%) creates downside headline friction.")

    if vol_anom >= 1.5:
        commentary.append(f"Elevated volume ({vol_anom:.2f}x 20d SMA) reflects high institutional attention.")
    elif vol_anom <= 0.8:
        commentary.append("Subdued trading volume indicates potential lack of conviction.")

    if mom_5d > 0.02:
        commentary.append(f"Pre-event price momentum is bullish (+{mom_5d*100:.1f}% over 5d).")
    elif mom_5d < -0.02:
        commentary.append(f"Pre-event price momentum is bearish ({mom_5d*100:.1f}% over 5d).")

    risk_text = " ".join(commentary) if commentary else "Features within historical normative baseline."

    border = "=" * 74
    sub_border = "-" * 74

    print(f"\n{border}")
    print(f"  SCENARIO: {scenario_title.upper()}")
    print(f"{border}")
    print(f"  Target           : {TARGET_NAME}")
    print(f"  Ticker / Event   : {ticker} | Type: {event_type} | Surprise: {surprise}")
    print(f"  Key Features     : Sent+(pos={pos_sent:.2f}, neg={neg_sent:.2f}) | Vol Anom: {vol_anom:.2f}x | Mom 5d: {mom_5d*100:+.2f}%")
    print(f"{sub_border}")
    print(f"  Confidence Score : P(Positive AR3) = {prob_pos*100:.2f}%  [Threshold = {threshold*100:.1f}%]")
    print(f"  Binary Signal    : [{decision_tag}]")
    print(f"  Thesis / Summary : {decision_desc}")
    print(f"  Risk Context     : {risk_text}")
    print(f"{border}\n")


# ---------------------------------------------------------------------------
# Pre-defined Mock Scenarios
# ---------------------------------------------------------------------------

def get_predefined_scenarios() -> List[Tuple[str, Dict[str, Any]]]:
    """Returns the 3 required evaluation scenarios."""
    return [
        (
            "Scenario A: High-Conviction Bullish (Earnings Beat)",
            {
                "ticker": "INFY",
                "event_type": "EARNINGS",
                "surprise": "BEAT",
                "finbert_positive": 0.92,
                "finbert_negative": 0.02,
                "finbert_neutral": 0.06,
                "volume_anomaly": 2.10,
                "momentum_5d": 0.035,
                "momentum_20d": 0.065,
                "volatility_20d": 0.012,
                "market_return_5d": 0.010,
                "magnitude": 0.85,
            },
        ),
        (
            "Scenario B: Strong Bearish / Headwind (Regulatory Miss)",
            {
                "ticker": "TCS",
                "event_type": "REGULATION",
                "surprise": "MISS",
                "finbert_positive": 0.04,
                "finbert_negative": 0.88,
                "finbert_neutral": 0.08,
                "volume_anomaly": 1.75,
                "momentum_5d": -0.042,
                "momentum_20d": -0.075,
                "volatility_20d": 0.028,
                "market_return_5d": -0.005,
                "magnitude": 0.70,
            },
        ),
        (
            "Scenario C: Neutral Macro Noise (In-Line Data)",
            {
                "ticker": "HDFCBANK",
                "event_type": "MACRO_DATA",
                "surprise": "IN_LINE",
                "finbert_positive": 0.05,
                "finbert_negative": 0.05,
                "finbert_neutral": 0.90,
                "volume_anomaly": 0.95,
                "momentum_5d": 0.002,
                "momentum_20d": 0.005,
                "volatility_20d": 0.014,
                "market_return_5d": 0.001,
                "magnitude": 0.10,
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Interactive Mode Prompts
# ---------------------------------------------------------------------------

def run_interactive_prompt(
    pipeline: Any,
    expected_cols: List[str],
    threshold: float,
) -> None:
    """Collects feature values interactively from the user via terminal prompt."""
    print("\n" + "=" * 74)
    print("  MARKETLENS LIVE INFERENCE — INTERACTIVE TEST MODE")
    print("=" * 74)
    print("Enter event parameters below (Press Enter to accept defaults):\n")

    def prompt(msg: str, default: str) -> str:
        val = input(f"  {msg} [{default}]: ").strip()
        return val if val else default

    def prompt_float(msg: str, default: float) -> float:
        val_str = prompt(msg, str(default))
        try:
            return float(val_str)
        except ValueError:
            print(f"    Invalid input '{val_str}', using default {default}")
            return default

    ticker = prompt("Ticker Symbol (e.g., INFY, TCS, RELIANCE)", "INFY").upper()

    print(f"\n  Available Event Types: {', '.join(EVENT_TYPES)}")
    event_type = prompt("Event Type", "EARNINGS").upper()
    if event_type not in EVENT_TYPES:
        print(f"    Warning: '{event_type}' not in standard taxonomy; setting to OTHER.")
        event_type = "OTHER"

    print(f"\n  Available Surprise Categories: {', '.join(SURPRISE_TYPES)}")
    surprise = prompt("Surprise", "BEAT").upper()
    if surprise not in SURPRISE_TYPES:
        surprise = "NOT_APPLICABLE"

    finbert_pos = prompt_float("FinBERT Positive Probability [0.0 - 1.0]", 0.85)
    finbert_neg = prompt_float("FinBERT Negative Probability [0.0 - 1.0]", 0.05)
    finbert_neu = max(0.0, 1.0 - (finbert_pos + finbert_neg))

    volume_anom = prompt_float("Volume Anomaly (Ratio vs 20-day SMA, e.g. 1.5)", 1.50)
    mom_5d_pct = prompt_float("5-Day Stock Return % (e.g., +2.5 or -1.5)", 2.0)
    mom_20d_pct = prompt_float("20-Day Stock Return %", 4.0)
    vol_20d = prompt_float("20-Day Daily Return Volatility (e.g. 0.015)", 0.015)
    mkt_5d_pct = prompt_float("5-Day NIFTY 50 Return %", 0.5)

    user_inputs = {
        "ticker": ticker,
        "event_type": event_type,
        "surprise": surprise,
        "finbert_positive": finbert_pos,
        "finbert_negative": finbert_neg,
        "finbert_neutral": finbert_neu,
        "volume_anomaly": volume_anom,
        "momentum_5d": mom_5d_pct / 100.0,
        "momentum_20d": mom_20d_pct / 100.0,
        "volatility_20d": vol_20d,
        "market_return_5d": mkt_5d_pct / 100.0,
        "magnitude": 0.5,
    }

    try:
        feature_df = build_scenario_dataframe(user_inputs, expected_cols)
        pred_class, prob_pos = run_prediction(pipeline, feature_df, threshold)
        print_prediction_card(
            f"Interactive Test: {ticker} ({event_type})",
            user_inputs,
            pred_class,
            prob_pos,
            threshold,
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to run prediction: {e}")


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MarketLens Model C (Event Classifier) Live Inference Tester"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Path to serialized joblib model artifact",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Classification decision threshold for positive AR3 (default: 0.50)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal prompt to test custom event features",
    )
    args = parser.parse_args()

    print("\n" + "=" * 74)
    print("  MARKETLENS PHASE 3 — MODEL C LIVE INFERENCE VERIFICATION")
    print("=" * 74)
    print(f"  Artifact Path : {args.model_path}")

    # 1. Load & Validate Artifact
    try:
        pipeline, expected_cols, metadata = load_and_validate_artifact(args.model_path)
    except Exception as err:
        print(f"\n[FATAL] Error loading model artifact:\n  {err}\n")
        sys.exit(1)

    print(f"  Pipeline Type : {type(pipeline).__name__}")
    print(f"  Expected Cols : {len(expected_cols)} features")
    if metadata.get("train_date_max"):
        print(f"  Trained Up To : {metadata['train_date_max']}")
    print(f"  Feature List  : {', '.join(expected_cols[:6])} ... (+{len(expected_cols)-6} more)")

    # 2. Interactive or Batch Scenario Mode
    if args.interactive:
        run_interactive_prompt(pipeline, expected_cols, args.threshold)
    else:
        print(f"\nRunning 3 built-in quantitative verification scenarios (Threshold={args.threshold:.2f})...")
        scenarios = get_predefined_scenarios()

        passed_count = 0
        for title, raw_inputs in scenarios:
            try:
                feature_df = build_scenario_dataframe(raw_inputs, expected_cols)
                pred_class, prob_pos = run_prediction(pipeline, feature_df, args.threshold)
                print_prediction_card(title, raw_inputs, pred_class, prob_pos, args.threshold)
                passed_count += 1
            except Exception as e:
                print(f"\n[ERROR] Failed running {title}: {e}")

        print("=" * 74)
        print(f"  VERIFICATION COMPLETE: {passed_count}/{len(scenarios)} scenarios executed successfully.")
        print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
