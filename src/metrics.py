"""
Industrial evaluation metrics for Mobile Game LTV models.
Implements:
1. RMSE (Root Mean Squared Error)
2. MAE (Mean Absolute Error)
3. Normalized Gini Coefficient
4. Top-k% Revenue Recall (e.g., Top-1%, Top-5%, Top-10%, Top-20%)
5. Decile Lift Table (marketing targeting efficiency)
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error."""
    y_pred = np.maximum(y_pred, 0.0)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error."""
    y_pred = np.maximum(y_pred, 0.0)
    return float(np.mean(np.abs(y_true - y_pred)))


def calc_raw_gini(actual: np.ndarray, pred: np.ndarray) -> float:
    """
    Computes raw Gini coefficient based on the Lorenz curve.
    Sorts by predicted value descending (tie-broken by actual descending),
    and measures the area between the cumulative actual revenue curve and the equality diagonal.
    """
    assert len(actual) == len(pred), "Actual and predicted arrays must have identical length."
    n = len(actual)
    total_actual = np.sum(actual)
    if total_actual == 0:
        return 0.0

    # Sort descending by prediction, tie-break by actual
    sort_order = np.lexsort((-actual, -pred))
    sorted_actual = actual[sort_order]

    cum_actual = np.cumsum(sorted_actual) / total_actual
    cum_pop = np.arange(1, n + 1) / n

    # Lorenz curve area between curve and diagonal
    gini = np.sum(cum_actual - cum_pop) / n
    return float(gini)


def calc_normalized_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Normalized Gini coefficient (Gini normalized by theoretical maximum).
    Normalized Gini = Gini(y_true, y_pred) / Gini(y_true, y_true).
    Ranges from 0.0 (random ranking) to 1.0 (perfect ranking).
    """
    model_gini = calc_raw_gini(y_true, y_pred)
    perfect_gini = calc_raw_gini(y_true, y_true)
    if perfect_gini == 0:
        return 0.0
    return float(model_gini / perfect_gini)


def calc_top_k_revenue_recall(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.10) -> float:
    """
    Computes the fraction of total actual revenue captured by the top k% of users
    with the highest predicted LTV.
    
    Formula:
        Recall@k = sum(actual revenue of top k% predicted users) / sum(all actual revenue)
    """
    total_rev = np.sum(y_true)
    if total_rev == 0:
        return 0.0
    n = len(y_true)
    n_top = max(1, int(np.round(n * k)))

    # Indices of top n predictions
    top_indices = np.argsort(y_pred)[-n_top:]
    top_actual_rev = np.sum(y_true[top_indices])
    return float(top_actual_rev / total_rev)


def calc_top_k_recall(y_true: np.ndarray, y_pred: np.ndarray, top_percent: float = 0.10) -> float:
    """Returns top k% revenue recall as a percentage [0, 100]."""
    return float(calc_top_k_revenue_recall(y_true, y_pred, k=top_percent) * 100.0)


def calc_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Spearman rank correlation."""
    from scipy.stats import spearmanr
    return float(spearmanr(y_true, y_pred).statistic)


def generate_decile_lift_table(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """
    Constructs a Decile Lift Table commonly used in gaming marketing & UA targeting.
    Divides users into deciles (Decile 1 = top 10% highest predicted LTV, Decile 10 = lowest).
    
    Columns:
    - decile: Decile rank (1 to 10)
    - user_count: Number of users in bin
    - pred_mean: Mean predicted LTV in decile
    - actual_mean: Mean actual LTV in decile
    - actual_sum: Total actual revenue in decile
    - revenue_share: Percentage of total dataset revenue captured by decile
    - cum_revenue_share: Cumulative percentage of revenue captured (Lorenz curve points)
    - lift: Ratio of decile actual mean to overall population mean
    """
    df = pd.DataFrame({"actual": y_true, "pred": np.maximum(y_pred, 0.0)})
    df = df.sort_values(by=["pred", "actual"], ascending=[False, False]).reset_index(drop=True)

    n = len(df)
    bin_size = n / n_bins
    df["decile"] = np.floor(df.index / bin_size).astype(int) + 1
    df.loc[df["decile"] > n_bins, "decile"] = n_bins

    total_actual = df["actual"].sum()
    pop_mean = df["actual"].mean()

    records = []
    cum_share = 0.0
    for d in range(1, n_bins + 1):
        subset = df[df["decile"] == d]
        cnt = len(subset)
        if cnt == 0:
            continue
        act_sum = subset["actual"].sum()
        act_mean = subset["actual"].mean()
        pred_mean = subset["pred"].mean()
        share = act_sum / total_actual if total_actual > 0 else 0.0
        cum_share += share
        lift = act_mean / pop_mean if pop_mean > 0 else 0.0

        records.append({
            "decile": d,
            "user_count": cnt,
            "pred_mean": pred_mean,
            "actual_mean": act_mean,
            "actual_sum": act_sum,
            "revenue_share": share,
            "cum_revenue_share": cum_share,
            "lift": lift,
        })

    return pd.DataFrame(records)


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict:
    """
    Runs full comprehensive evaluation suite and returns formatted metrics dict.
    """
    y_pred = np.maximum(y_pred, 0.0)
    rmse = calc_rmse(y_true, y_pred)
    mae = calc_mae(y_true, y_pred)
    norm_gini = calc_normalized_gini(y_true, y_pred)

    recall_1 = calc_top_k_revenue_recall(y_true, y_pred, k=0.01)
    recall_5 = calc_top_k_revenue_recall(y_true, y_pred, k=0.05)
    recall_10 = calc_top_k_revenue_recall(y_true, y_pred, k=0.10)
    recall_20 = calc_top_k_revenue_recall(y_true, y_pred, k=0.20)

    # Spearman rank correlation
    from scipy.stats import spearmanr, pearsonr
    spearman_corr = float(spearmanr(y_true, y_pred).statistic)
    pearson_corr = float(pearsonr(y_true, y_pred).statistic)

    lift_table = generate_decile_lift_table(y_true, y_pred)

    metrics = {
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "normalized_gini": norm_gini,
        "top_1%_revenue_recall": recall_1,
        "top_5%_revenue_recall": recall_5,
        "top_10%_revenue_recall": recall_10,
        "top_20%_revenue_recall": recall_20,
        "spearman_correlation": spearman_corr,
        "pearson_correlation": pearson_corr,
        "decile_lift_table": lift_table.to_dict(orient="records"),
    }
    return metrics
