"""
scripts/run_phase3_training.py — MarketLens Phase 3 CLI Entry Point
=====================================================================
Runs the complete Phase 3 Financial ML pipeline:
  1. Builds the event matrix from PostgreSQL (events + market_data + indices)
  2. Engineers technical and event features (no look-ahead bias)
  3. Performs chronological split (70/15/15)
  4. Trains 3-model ablation experiment (Market Baseline / + Sentiment / Event-Aware)
  5. Evaluates on out-of-sample test set
  6. Exports Model C to models/event_classifier_v1.joblib

Usage:
    python scripts/run_phase3_training.py [--log-level DEBUG|INFO|WARNING]
"""
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ml.dataset_builder import (
    build_event_matrix,
    _load_market_data,
    _load_index_data,
)
from backend.ml.features import build_features, TARGET_COL
from backend.ml.train import train_all_models
from database.connection import SessionLocal


def parse_args():
    parser = argparse.ArgumentParser(description="MarketLens Phase 3 - Financial ML Training")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("phase3")

    logger.info("=" * 60)
    logger.info("MarketLens Phase 3 — Financial ML Pipeline")
    logger.info("=" * 60)

    # Step 1: Build event matrix
    logger.info("\n[Step 1/4] Building event matrix from PostgreSQL...")
    with SessionLocal() as session:
        event_df = build_event_matrix(session)
        mkt_df   = _load_market_data(session)
        idx_df   = _load_index_data(session)

    if event_df.empty:
        logger.error(
            "\nEvent matrix is EMPTY. No events with valid market data found.\n"
            "Ensure you have:\n"
            "  1. Populated the events table: python scripts/extract_events.py --limit 100\n"
            "  2. Populated market_data: python scripts/collect_market_data.py\n"
            "  3. Your tickers in events match company tickers in market_data"
        )
        sys.exit(1)

    logger.info("Event matrix: %d rows", len(event_df))
    logger.info("Target distribution:\n%s", event_df[TARGET_COL].value_counts().to_string())

    # Step 2: Feature engineering
    logger.info("\n[Step 2/4] Engineering features...")
    feature_df = build_features(event_df, mkt_df, idx_df)
    logger.info(
        "Feature matrix: %d rows x %d columns",
        feature_df.shape[0], feature_df.shape[1],
    )

    if len(feature_df) < 20:
        logger.error(
            "Fewer than 20 samples after feature engineering (%d). "
            "Cannot train reliably. Run more data collection first.",
            len(feature_df),
        )
        sys.exit(1)

    # Step 3 + 4: Train ablation models
    logger.info("\n[Step 3/4] Training 3-model ablation experiment...")
    results = train_all_models(feature_df)

    # Summary
    logger.info("\n[Step 4/4] Generating summary report...")
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 COMPLETE")
    logger.info("=" * 60)
    for model_name, metrics_list in results["metrics"].items():
        test_m = next((m for m in metrics_list if m["split"] == "test"), None)
        if test_m:
            logger.info(
                "%-35s | Acc: %.4f | F1: %.4f | AUC: %s",
                model_name,
                test_m["accuracy"],
                test_m["macro_f1"],
                test_m["roc_auc"],
            )

    logger.info(
        "\nArtifact: models/event_classifier_v1.joblib"
        "\nNext Step: Open notebooks/03_model_comparison.ipynb for visual analysis"
    )


if __name__ == "__main__":
    main()
