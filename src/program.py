"""
Seed program for AlphaEvolve Mobile Game LTV Feature Engineering Optimization.
The EVOLVE-BLOCK contains the `engineer_features(df)` function.
AlphaEvolve will evolve this function to discover more predictive interaction,
aggregation, and momentum features for 180-day LTV forecasting.
"""

from typing import Any, Mapping
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
_ROOT = str(Path.cwd())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


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


def evaluate(eval_inputs: Mapping[str, Any] = None) -> dict[str, float]:
    """
    Evaluates the engineered features by training a LightGBM Tweedie baseline
    on the training split (60%) and scoring on the evaluation split (20%).
    Returns a dictionary of metrics for AlphaEvolve.
    """
    from src.evaluate import evaluate_feature_function

    return evaluate_feature_function(engineer_features)


if __name__ == "__main__":
    print("Testing seed program evaluate() locally...")
    res = evaluate({})
    print("Evaluation result:", res)
