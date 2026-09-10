"""
Unit tests for 3-way dataset splitting and stratification consistency.
"""

import numpy as np
import pandas as pd
import pytest

from src.dataset import engineer_features, split_train_eval_holdout


@pytest.fixture
def dummy_dataset() -> pd.DataFrame:
    """Generates synthetic wide table mimicking train_wide.csv."""
    rng = np.random.RandomState(42)
    n = 1000
    df = pd.DataFrame(
        {
            "user_id": [f"user_{i}" for i in range(n)],
            "platform": rng.choice(["android", "ios"], size=n, p=[0.7, 0.3]),
            "country_tier": rng.choice(["T1", "T2", "T3"], size=n),
            "channel_tier": rng.choice(["organic", "paid"], size=n),
            "revenue_d0": rng.exponential(0.5, size=n),
            "revenue_d1": rng.exponential(0.3, size=n),
            "revenue_d2": rng.exponential(0.2, size=n),
            "revenue_d5": rng.exponential(0.1, size=n),
            "revenue_d6": rng.exponential(0.1, size=n),
            "revenue_d7": rng.exponential(0.1, size=n),
            "total_revenue_d7": rng.exponential(1.5, size=n),
            "ad_revenue_usd_sum": rng.exponential(0.5, size=n),
            "iap_revenue_usd_sum": rng.exponential(1.0, size=n),
            "ad_impression_count": rng.poisson(10, size=n),
            "total_events": rng.poisson(50, size=n),
            "active_days_count": rng.randint(1, 9, size=n),
            "install_week": rng.randint(1, 8, size=n),
            "ltv_d8_d180": rng.exponential(2.0, size=n) * (rng.rand(n) > 0.8),
            "is_payer_d8_d180": np.zeros(n),
        }
    )
    df["is_payer_d8_d180"] = (df["ltv_d8_d180"] > 0).astype(int)
    return df


def test_feature_engineering_integrity(dummy_dataset):
    df_eng = engineer_features(dummy_dataset)
    assert "early_rev_d0_d2" in df_eng.columns
    assert "late_rev_d5_d7" in df_eng.columns
    assert "rev_growth_ratio" in df_eng.columns
    assert "ad_rev_per_imp" in df_eng.columns
    assert "events_per_active_day" in df_eng.columns
    assert len(df_eng) == len(dummy_dataset)


def test_split_train_eval_holdout_proportions(dummy_dataset):
    train_df, eval_df, holdout_df = split_train_eval_holdout(
        dummy_dataset,
        train_ratio=0.60,
        eval_ratio=0.20,
        holdout_ratio=0.20,
        random_state=42,
        method="stratified",
    )
    total = len(dummy_dataset)
    assert len(train_df) + len(eval_df) + len(holdout_df) == total
    assert abs(len(train_df) / total - 0.60) <= 0.01
    assert abs(len(eval_df) / total - 0.20) <= 0.01
    assert abs(len(holdout_df) / total - 0.20) <= 0.01

    # Check non-overlapping user IDs
    s_train = set(train_df["user_id"])
    s_eval = set(eval_df["user_id"])
    s_holdout = set(holdout_df["user_id"])
    assert len(s_train.intersection(s_eval)) == 0
    assert len(s_train.intersection(s_holdout)) == 0
    assert len(s_eval.intersection(s_holdout)) == 0
