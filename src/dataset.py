"""
Data loading, feature engineering, and train/eval/holdout splitting pipeline.
Supports:
- Scientific multi-factor stratified 3-way splitting (train: 60%, eval: 20%, holdout: 20%)
- Temporal Out-Of-Time (OOT) cohort splitting
- Seed feature engineering marked with EVOLVE-BLOCK
"""

from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Loads CSV wide table into a pandas DataFrame."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df


# EVOLVE-BLOCK-START
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific interaction and momentum features for mobile game LTV:
    1. Early vs Late Revenue Momentum:
       - early_rev_d0_d2: sum of revenue on D0, D1, D2
       - late_rev_d5_d7: sum of revenue on D5, D6, D7
       - rev_growth_ratio: late / (early + 0.01) - monetisation acceleration
    2. Engagement & Monetization Efficiencies:
       - ad_rev_per_imp: ad revenue per impression
       - events_per_active_day: event density per active day
       - active_day_ratio: active days / 8.0
       - iap_share: ratio of IAP revenue to total D7 revenue
    """
    df = df.copy()

    # Revenue momentum across observation window
    df["early_rev_d0_d2"] = df["revenue_d0"] + df["revenue_d1"] + df["revenue_d2"]
    df["late_rev_d5_d7"] = df["revenue_d5"] + df["revenue_d6"] + df["revenue_d7"]
    df["rev_growth_ratio"] = (df["late_rev_d5_d7"] + 0.01) / (df["early_rev_d0_d2"] + 0.01)

    # Unit metrics
    df["ad_rev_per_imp"] = df["ad_revenue_usd_sum"] / (df["ad_impression_count"] + 1e-4)
    df["events_per_active_day"] = df["total_events"] / (df["active_days_count"] + 1e-4)
    df["active_day_ratio"] = df["active_days_count"] / 8.0
    df["iap_share"] = df["iap_revenue_usd_sum"] / (df["total_revenue_d7"] + 1e-4)

    return df
# EVOLVE-BLOCK-END


def _create_composite_strata(df: pd.DataFrame) -> np.ndarray:
    """Creates a robust composite stratification key based on target LTV, revenue, and platform."""
    y = df["ltv_d8_d180"].values
    payer_mask = y > 0

    # Payer deciles (10 bins for payers, bin 0 for non-payers -> 11 strata)
    ltv_strata = np.zeros(len(df), dtype=int)
    if np.any(payer_mask):
        payer_quantiles = np.percentile(y[payer_mask], np.linspace(10, 90, 9))
        ltv_strata[payer_mask] = np.digitize(y[payer_mask], payer_quantiles) + 1

    # Revenue tiers (0, (0, 0.5], (0.5, 2.0], (2.0, 10.0], >10.0)
    rev_strata = np.digitize(df["total_revenue_d7"].values, [0.0, 0.5, 2.0, 10.0])

    # Platform code (0 for android, 1 for ios)
    plat_code = (df["platform"] == "ios").astype(int).values

    # Composite strata key
    composite = ltv_strata * 20 + rev_strata * 2 + plat_code
    counts = Counter(composite)
    for k, v in counts.items():
        if v < 3:
            composite[composite == k] = 0
    return composite


def split_train_eval_holdout(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    eval_ratio: float = 0.20,
    holdout_ratio: float = 0.20,
    random_state: int = 42,
    method: str = "stratified",
    oot_eval_week: int = 4,
    oot_holdout_week: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into 3 distribution-consistent parts:
    - Train set (default 60%): Model fitting for candidate feature variants
    - Eval set  (default 20%): Evolutionary candidate selection and champion picking
    - Holdout set (default 20%): Final evaluation comparing champion code vs seed code

    Methods:
    - 'stratified': Multi-factor stratified split preserving LTV deciles,
      early revenue tiers, and platform proportions across all 3 sets.
    - 'oot': Temporal Out-Of-Time cohort split based on `install_week`.
    """
    df = df.copy()
    total_len = len(df)

    if method == "oot":
        print(
            f"Splitting data using Out-Of-Time (OOT) cohort split "
            f"(train: <{oot_eval_week}, eval: [{oot_eval_week}, {oot_holdout_week}), holdout: >={oot_holdout_week})..."
        )
        train_df = df[df["install_week"] < oot_eval_week].copy().reset_index(drop=True)
        eval_df = df[(df["install_week"] >= oot_eval_week) & (df["install_week"] < oot_holdout_week)].copy().reset_index(drop=True)
        holdout_df = df[df["install_week"] >= oot_holdout_week].copy().reset_index(drop=True)

    elif method == "stratified":
        print(
            f"Splitting data using 3-way multi-factor stratified split "
            f"(train={train_ratio:.1%}, eval={eval_ratio:.1%}, holdout={holdout_ratio:.1%}, seed={random_state})..."
        )
        composite = _create_composite_strata(df)

        # Step 1: Split holdout out of total
        sss_holdout = StratifiedShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=random_state)
        dummy_x = np.zeros(total_len)
        train_eval_idx, holdout_idx = next(sss_holdout.split(dummy_x, composite))

        # Step 2: Split remaining train_eval into train and eval
        train_eval_df = df.iloc[train_eval_idx]
        train_eval_composite = composite[train_eval_idx]

        eval_rel_ratio = eval_ratio / (train_ratio + eval_ratio)
        sss_eval = StratifiedShuffleSplit(n_splits=1, test_size=eval_rel_ratio, random_state=random_state)
        dummy_te = np.zeros(len(train_eval_df))
        train_sub_idx, eval_sub_idx = next(sss_eval.split(dummy_te, train_eval_composite))

        train_idx = train_eval_idx[train_sub_idx]
        eval_idx = train_eval_idx[eval_sub_idx]

        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        eval_df = df.iloc[eval_idx].copy().reset_index(drop=True)
        holdout_df = df.iloc[holdout_idx].copy().reset_index(drop=True)
    else:
        raise ValueError(f"Unknown split method: {method}. Choose 'stratified' or 'oot'.")

    print(
        f"Train set:   {len(train_df):,} users ({len(train_df)/total_len:.2%})\n"
        f"Eval set:    {len(eval_df):,} users ({len(eval_df)/total_len:.2%})\n"
        f"Holdout set: {len(holdout_df):,} users ({len(holdout_df)/total_len:.2%})"
    )
    return train_df, eval_df, holdout_df


def split_train_holdout(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    method: str = "stratified",
    oot_week_threshold: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy 2-way split wrapper for backward compatibility."""
    train_ratio = 1.0 - test_size
    train_df, _, holdout_df = split_train_eval_holdout(
        df,
        train_ratio=train_ratio,
        eval_ratio=0.0,
        holdout_ratio=test_size,
        random_state=random_state,
        method=method,
    )
    return train_df, holdout_df


def save_splits(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    train_path: str | Path,
    holdout_path: str | Path,
    eval_df: Optional[pd.DataFrame] = None,
    eval_path: Optional[str | Path] = None,
):
    """Saves train, eval (optional), and holdout splits to CSV files."""
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(holdout_path).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)
    print(f"Saved train split   -> {train_path} ({len(train_df):,} rows)")
    if eval_df is not None and eval_path is not None:
        Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_csv(eval_path, index=False)
        print(f"Saved eval split    -> {eval_path} ({len(eval_df):,} rows)")
    print(f"Saved holdout split -> {holdout_path} ({len(holdout_df):,} rows)")


def prepare_features(
    df: pd.DataFrame,
    cat_cols: List[str],
    drop_cols: List[str],
    target_col: str = "ltv_d8_d180",
    id_col: str = "user_id",
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepares feature matrix X, target array y, and ID array.
    Ensures categorical columns are typed as pandas 'category'.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in drop_cols]

    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    X = df[feature_cols]
    y = df[target_col].values if target_col in df.columns else None
    user_ids = df[id_col].values if id_col in df.columns else None

    return X, y, user_ids
