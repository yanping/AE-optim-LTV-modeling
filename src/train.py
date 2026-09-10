"""
Training pipeline for Mobile Game LTV LightGBM Tweedie Model.
Handles:
- Dataset loading and stratified/OOT splitting
- Feature preparation
- Training LightGBM Tweedie regression model (p=1.5) with early stopping
- Saving model artifact
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import DEFAULT_TWEEDIE_PARAMS, ProjectConfig
from src.dataset import (
    engineer_features,
    load_data,
    prepare_features,
    save_splits,
    split_train_holdout,
)
from src.models import LightGBMTweedieModel


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM Tweedie LTV forecasting model.")
    parser.add_argument(
        "--split-method",
        type=str,
        default="stratified",
        choices=["stratified", "oot"],
        help="Train/holdout split strategy (default: 'stratified')",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Holdout proportion (default: 0.20 for 20%%)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--re-split",
        action="store_true",
        help="Force re-splitting even if split files already exist",
    )
    parser.add_argument(
        "--tweedie-p",
        type=float,
        default=1.5,
        help="Tweedie variance power parameter (default: 1.5)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Boosting learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=31,
        help="Max tree leaves (default: 31)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Maximum boosting iterations (default: 500)",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=40,
        help="Early stopping rounds (default: 40)",
    )
    return parser.parse_args()


def run_training(args: argparse.Namespace, config: ProjectConfig) -> LightGBMTweedieModel:
    t0 = time.time()
    print("=" * 70)
    print("  MOBILE GAME LTV MODEL TRAINING PIPELINE (LightGBM Tweedie)")
    print("=" * 70)

    # 1. Dataset Preparation & Splitting
    if config.train_split_path.exists() and config.holdout_split_path.exists() and not args.re_split:
        print(f"Loading existing train/holdout splits:")
        print(f"  Train:   {config.train_split_path}")
        print(f"  Holdout: {config.holdout_split_path}")
        train_df = pd.read_csv(config.train_split_path)
        holdout_df = pd.read_csv(config.holdout_split_path)
    else:
        print(f"Loading full wide table from {config.raw_wide_path}...")
        raw_df = load_data(config.raw_wide_path)
        print("Applying feature engineering...")
        engineered_df = engineer_features(raw_df)

        train_df, holdout_df = split_train_holdout(
            engineered_df,
            test_size=args.test_size,
            random_state=args.random_state,
            method=args.split_method,
        )
        save_splits(train_df, holdout_df, config.train_split_path, config.holdout_split_path)

    # 2. Feature Extraction
    print("\nPreparing feature matrices...")
    X_train, y_train, train_ids = prepare_features(
        train_df, cat_cols=config.cat_cols, drop_cols=config.drop_cols, target_col=config.target_col, id_col=config.id_col
    )
    X_val, y_val, val_ids = prepare_features(
        holdout_df, cat_cols=config.cat_cols, drop_cols=config.drop_cols, target_col=config.target_col, id_col=config.id_col
    )
    print(f"Feature count: {X_train.shape[1]} features.")
    print(f"Train samples: {len(X_train):,} | Holdout samples: {len(X_val):,}")

    # 3. Model Training
    print(f"\n--- Training LightGBMTweedieModel (p={args.tweedie_p}) ---")
    model = LightGBMTweedieModel(
        tweedie_variance_power=args.tweedie_p,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping,
    )

    t_model = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    model.save(config.model_path)
    print(f"  Finished training in {time.time() - t_model:.2f}s")
    print(f"  Model saved to {config.model_path}")
    print(f"\nTraining pipeline completed in {time.time() - t0:.2f}s.")
    return model


def main():
    args = parse_train_args()
    config = ProjectConfig()
    run_training(args, config)


if __name__ == "__main__":
    main()
