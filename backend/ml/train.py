"""
backend/ml/train.py — MarketLens Phase 3
==========================================
3-Model Ablation Experiment: Does Event Intelligence add predictive edge?

Trains three scikit-learn pipelines on chronologically split data:
  Model A: Market Baseline    — technical features only
  Model B: Market + Sentiment — technical + FinBERT probabilities
  Model C: Event-Aware        — all features (technical + sentiment + event_type + surprise + magnitude)

Chronological split (NO shuffle, NO K-Fold to prevent look-ahead bias):
  Train : first 70% of events (sorted by event_date)
  Val   : next  15%
  Test  : final 15%  <- out-of-sample evaluation

Metrics: Accuracy, Macro F1, ROC-AUC, Precision/Recall for class 1.
Feature importances extracted from Model C to assess event vs technical weights.

Artifacts saved to models/:
  event_classifier_v1.joblib  — Model C pipeline + feature names (for Phase 5 agent)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.ml.dataset_builder import (
    _load_index_data,
    _load_market_data,
    build_event_matrix,
)
from backend.ml.features import (
    FEATURE_SET_A,
    FEATURE_SET_B,
    FEATURE_SET_C,
    TARGET_COL,
    build_features,
)
from database.connection import SessionLocal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_PATH = MODELS_DIR / "event_classifier_v1.joblib"


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically (NO shuffle).

    Data MUST be sorted by event_date before calling.
    Train: first train_frac | Val: next val_frac | Test: remainder.

    Returns (train_df, val_df, test_df)
    """
    n = len(df)
    if n < 10:
        raise ValueError(
            f"Dataset too small for chronological split (n={n}). "
            "Run more extract_events.py to populate the events table."
        )

    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_df = df.iloc[:n_train].copy()
    val_df   = df.iloc[n_train : n_train + n_val].copy()
    test_df  = df.iloc[n_train + n_val :].copy()

    logger.info(
        "Chronological split — Train: %d (%.0f%%) | Val: %d (%.0f%%) | Test: %d (%.0f%%)",
        len(train_df), 100 * len(train_df) / n,
        len(val_df),   100 * len(val_df)   / n,
        len(test_df),  100 * len(test_df)  / n,
    )
    logger.info(
        "Date ranges — Train: %s -> %s | Val: %s -> %s | Test: %s -> %s",
        train_df["event_date"].min(), train_df["event_date"].max(),
        val_df["event_date"].min(),   val_df["event_date"].max(),
        test_df["event_date"].min(),  test_df["event_date"].max(),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(feature_cols: list[str]) -> Pipeline:
    """
    Build a scikit-learn Pipeline with:
      - SimpleImputer (median) for all numeric features
      - StandardScaler
      - RandomForestClassifier (class_weight='balanced' for imbalanced targets)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale",  StandardScaler()),
                ]),
                feature_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
    feature_cols: list[str],
) -> dict:
    """Evaluate a fitted pipeline and return a metrics dictionary."""
    X_feat = X[feature_cols]
    y_pred = pipeline.predict(X_feat)
    y_prob = None
    try:
        y_prob = pipeline.predict_proba(X_feat)[:, 1]
    except Exception:
        pass

    acc   = accuracy_score(y, y_pred)
    f1    = f1_score(y, y_pred, average="macro", zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(
        y, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    auc = roc_auc_score(y, y_prob) if y_prob is not None and len(np.unique(y)) > 1 else float("nan")

    metrics = {
        "split":        split_name,
        "n_samples":    len(y),
        "accuracy":     round(acc, 4),
        "macro_f1":     round(f1, 4),
        "roc_auc":      round(auc, 4) if not np.isnan(auc) else None,
        "precision_1":  round(float(prec[1]), 4) if len(prec) > 1 else None,
        "recall_1":     round(float(rec[1]), 4)  if len(rec)  > 1 else None,
        "class_dist":   y.value_counts().to_dict(),
    }
    return metrics


def print_metrics_table(all_metrics: dict[str, list[dict]]) -> None:
    """Pretty-print a comparative metrics table."""
    header = f"\n{'='*80}\n  3-MODEL ABLATION RESULTS (Out-of-Sample Test Set)\n{'='*80}"
    print(header)
    row_fmt = "{:<12} {:<10} {:<10} {:<10} {:<14} {:<10}"
    print(row_fmt.format("Model", "Accuracy", "Macro F1", "ROC-AUC", "Precision@1", "Recall@1"))
    print("-" * 70)
    for model_name, metrics_list in all_metrics.items():
        test_m = next((m for m in metrics_list if m["split"] == "test"), None)
        if test_m:
            print(row_fmt.format(
                model_name,
                str(test_m["accuracy"]),
                str(test_m["macro_f1"]),
                str(test_m["roc_auc"] or "N/A"),
                str(test_m["precision_1"] or "N/A"),
                str(test_m["recall_1"] or "N/A"),
            ))
    print("=" * 80)


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def extract_feature_importances(
    pipeline: Pipeline,
    feature_cols: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract and return sorted feature importances from Model C."""
    clf = pipeline.named_steps["classifier"]
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    return fi_df.head(top_n)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_all_models(df: pd.DataFrame) -> dict:
    """
    Run the full 3-model ablation experiment.

    Returns
    -------
    dict with keys:
      "pipelines": {model_name: fitted_pipeline}
      "metrics":   {model_name: [train_metrics, val_metrics, test_metrics]}
      "feature_importances": DataFrame (Model C)
      "feature_cols_C": list[str]
      "split_dates": dict with train/val/test date ranges
    """
    # Sort by event date (critical for chronological split)
    df = df.sort_values("event_date").reset_index(drop=True)

    # Drop rows where target is missing
    df = df[df[TARGET_COL].notna()].copy()

    if len(df) < 20:
        raise RuntimeError(
            f"Only {len(df)} valid events after feature engineering. "
            "Need at least 20. Please run extract_events.py to populate more events."
        )

    # Chronological split
    train_df, val_df, test_df = chronological_split(df)

    models = {
        "Model A (Market Baseline)":    FEATURE_SET_A,
        "Model B (Market + Sentiment)": FEATURE_SET_B,
        "Model C (Event-Aware)":        FEATURE_SET_C,
    }

    pipelines = {}
    all_metrics: dict[str, list[dict]] = {}

    for model_name, feature_cols in models.items():
        logger.info("\n%s\nTraining %s (%d features)...", "="*60, model_name, len(feature_cols))

        # Filter to features that actually exist in the DataFrame
        available_feats = [f for f in feature_cols if f in df.columns]
        missing_feats   = set(feature_cols) - set(available_feats)
        if missing_feats:
            logger.warning("Missing features for %s: %s", model_name, missing_feats)

        if not available_feats:
            logger.error("No features available for %s — skipping.", model_name)
            continue

        pipeline = build_pipeline(available_feats)

        X_train = train_df[available_feats]
        y_train = train_df[TARGET_COL].astype(int)
        X_val   = val_df[available_feats]
        y_val   = val_df[TARGET_COL].astype(int)
        X_test  = test_df[available_feats]
        y_test  = test_df[TARGET_COL].astype(int)

        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        logger.info("  Training completed in %.2fs", train_time)

        # Evaluate on all splits
        metrics = []
        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("val",   X_val,   y_val),
            ("test",  X_test,  y_test),
        ]:
            m = evaluate_model(pipeline, X_split.to_frame() if X_split.ndim == 1 else X_split,
                               y_split, split_name, available_feats)
            metrics.append(m)
            logger.info(
                "  [%s] Acc=%.4f | F1=%.4f | AUC=%s | P@1=%s | R@1=%s",
                split_name.upper(),
                m["accuracy"], m["macro_f1"],
                m["roc_auc"], m["precision_1"], m["recall_1"],
            )

        pipelines[model_name]   = pipeline
        all_metrics[model_name] = metrics

    print_metrics_table(all_metrics)

    # Feature importances for Model C
    fi_df = pd.DataFrame()
    model_c_key = "Model C (Event-Aware)"
    model_c_feats = [f for f in FEATURE_SET_C if f in df.columns]
    if model_c_key in pipelines and model_c_feats:
        fi_df = extract_feature_importances(pipelines[model_c_key], model_c_feats)
        print(f"\n{'='*60}")
        print("  MODEL C — TOP 20 FEATURE IMPORTANCES")
        print("="*60)
        print(fi_df.to_string(index=False))

    # Save Model C artifact for Phase 5 inference
    if model_c_key in pipelines:
        artifact = {
            "pipeline":      pipelines[model_c_key],
            "feature_cols":  model_c_feats,
            "target_col":    TARGET_COL,
            "metrics":       all_metrics.get(model_c_key, []),
            "train_date_max": str(train_df["event_date"].max()),
        }
        joblib.dump(artifact, ARTIFACT_PATH)
        logger.info("\nModel C artifact saved to: %s", ARTIFACT_PATH)

    return {
        "pipelines":           pipelines,
        "metrics":             all_metrics,
        "feature_importances": fi_df,
        "feature_cols_C":      model_c_feats,
        "train_df":            train_df,
        "val_df":              val_df,
        "test_df":             test_df,
        "split_dates": {
            "train": (str(train_df["event_date"].min()), str(train_df["event_date"].max())),
            "val":   (str(val_df["event_date"].min()),   str(val_df["event_date"].max())),
            "test":  (str(test_df["event_date"].min()),  str(test_df["event_date"].max())),
        },
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Phase 3: Building event matrix from database...")
    with SessionLocal() as session:
        event_df = build_event_matrix(session)
        mkt_df   = _load_market_data(session)
        idx_df   = _load_index_data(session)

    if event_df.empty:
        logger.error(
            "Event matrix is empty! Make sure you have run:\n"
            "  python scripts/extract_events.py --limit 50\n"
            "to populate the events table before training."
        )
        sys.exit(1)

    logger.info("Building features...")
    feature_df = build_features(event_df, mkt_df, idx_df)

    logger.info(
        "Feature matrix ready: %d rows, %d columns",
        len(feature_df), len(feature_df.columns),
    )
    logger.info("Target distribution:\n%s", feature_df[TARGET_COL].value_counts().to_string())

    results = train_all_models(feature_df)

    logger.info("\n=== Phase 3 Training Complete ===")
    logger.info("Artifact: %s", ARTIFACT_PATH)
