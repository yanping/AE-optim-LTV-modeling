"""
Local candidate evaluation harness for AlphaEvolve LTV Feature Engineering Evolution.

Responsibilities:
1. Load and cache the 60% train and 20% eval datasets in memory.
2. Dynamically execute candidate feature engineering code in a protected namespace.
3. Train the frozen LightGBM Tweedie baseline on the transformed train features.
4. Predict on the evaluation split and compute validation metrics (Normalized Gini, Top-10% Recall, etc.).
5. Provide structured evaluation responses with scores and diagnostic insights for AlphaEvolve.
"""

from collections.abc import Mapping
import logging
from pathlib import Path
import threading
import traceback
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DEFAULT_TWEEDIE_PARAMS, ProjectConfig
from src.dataset import (
    load_data,
    prepare_features,
    save_splits,
    split_train_eval_holdout,
)
from src.metrics import calc_normalized_gini, calc_rmse, calc_spearman, calc_top_k_recall
from src.models import LightGBMTweedieModel

logger = logging.getLogger(__name__)

# Global dataset cache and thread lock
_DATA_LOCK = threading.Lock()
_CACHED_TRAIN_DF: Optional[pd.DataFrame] = None
_CACHED_EVAL_DF: Optional[pd.DataFrame] = None
_CACHED_HOLDOUT_DF: Optional[pd.DataFrame] = None


def get_cached_splits(
    config: Optional[ProjectConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, eval_df, holdout_df) from cache or loads/splits from disk.
    Thread-safe so parallel candidate evaluations can share the memory.
    """
    global _CACHED_TRAIN_DF, _CACHED_EVAL_DF, _CACHED_HOLDOUT_DF

    with _DATA_LOCK:
        if (
            _CACHED_TRAIN_DF is not None
            and _CACHED_EVAL_DF is not None
            and _CACHED_HOLDOUT_DF is not None
        ):
            return _CACHED_TRAIN_DF, _CACHED_EVAL_DF, _CACHED_HOLDOUT_DF

        if config is None:
            config = ProjectConfig()

        train_path = config.train_split_path
        eval_path = config.eval_split_path
        holdout_path = config.holdout_split_path

        # If splits already exist on disk, load them
        if train_path.exists() and eval_path.exists() and holdout_path.exists():
            logger.info("Loading cached splits from CSV files...")
            train_df = pd.read_csv(train_path)
            eval_df = pd.read_csv(eval_path)
            holdout_df = pd.read_csv(holdout_path)
        else:
            logger.info("Splits not found or incomplete. Generating 3-way split from train_wide.csv...")
            raw_df = load_data(config.raw_wide_path)
            train_df, eval_df, holdout_df = split_train_eval_holdout(
                raw_df,
                train_ratio=config.train_ratio,
                eval_ratio=config.eval_ratio,
                holdout_ratio=config.holdout_ratio,
                random_state=config.random_state,
                method=config.split_method,
            )
            save_splits(
                train_df,
                holdout_df,
                train_path,
                holdout_path,
                eval_df=eval_df,
                eval_path=eval_path,
            )

        _CACHED_TRAIN_DF = train_df
        _CACHED_EVAL_DF = eval_df
        _CACHED_HOLDOUT_DF = holdout_df

        return _CACHED_TRAIN_DF, _CACHED_EVAL_DF, _CACHED_HOLDOUT_DF


def _clean_engineered_dataframe(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """Cleans NaN, Inf, and ensures compatibility with LightGBM."""
    df = df.copy()
    # Check all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        vals = df[col].values
        mask_inf = np.isinf(vals)
        if np.any(mask_inf):
            median_val = np.nanmedian(vals[~mask_inf]) if np.any(~mask_inf) else 0.0
            df.loc[mask_inf, col] = median_val

        mask_nan = np.isnan(df[col].values)
        if np.any(mask_nan):
            median_val = np.nanmedian(df[col].values[~mask_nan]) if np.any(~mask_nan) else 0.0
            df[col] = df[col].fillna(median_val)

    return df


def evaluate_feature_function(
    engineer_func: Callable[[pd.DataFrame], pd.DataFrame],
    config: Optional[ProjectConfig] = None,
    eval_target: str = "eval",
) -> Dict[str, float]:
    """
    Applies the candidate feature function to the dataset, fits the LightGBM Tweedie model,
    and returns evaluation metrics.

    Args:
        engineer_func: The candidate feature engineering callable.
        config: Project configuration.
        eval_target: 'eval' (score on 20% eval set) or 'holdout' (score on 20% holdout set).
    """
    if config is None:
        config = ProjectConfig()

    train_raw, eval_raw, holdout_raw = get_cached_splits(config)
    test_raw = eval_raw if eval_target == "eval" else holdout_raw

    # 1. Apply feature engineering
    train_eng = engineer_func(train_raw.copy())
    test_eng = engineer_func(test_raw.copy())

    if not isinstance(train_eng, pd.DataFrame) or not isinstance(test_eng, pd.DataFrame):
        raise TypeError("engineer_features must return a pandas DataFrame.")

    # 2. Clean numeric anomalies
    train_clean = _clean_engineered_dataframe(train_eng, list(train_raw.columns))
    test_clean = _clean_engineered_dataframe(test_eng, list(test_raw.columns))

    # 3. Prepare feature matrices
    X_train, y_train, _ = prepare_features(
        train_clean,
        cat_cols=config.cat_cols,
        drop_cols=config.drop_cols,
        target_col=config.target_col,
        id_col=config.id_col,
    )
    X_test, y_test, _ = prepare_features(
        test_clean,
        cat_cols=config.cat_cols,
        drop_cols=config.drop_cols,
        target_col=config.target_col,
        id_col=config.id_col,
    )

    # Align column sets (if candidate generated conditional columns)
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    if len(common_cols) == 0:
        raise ValueError("No common feature columns found between train and test sets.")
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # 4. Fit LightGBM Tweedie Model (fixed hyperparameters)
    model = LightGBMTweedieModel(
        tweedie_variance_power=1.5,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    # 5. Predict and evaluate
    y_pred = model.predict(X_test)

    gini = calc_normalized_gini(y_test, y_pred)
    top10_recall = calc_top_k_recall(y_test, y_pred, top_percent=0.10)
    rmse = calc_rmse(y_test, y_pred)
    spearman = calc_spearman(y_test, y_pred)

    return {
        "normalized_gini": float(gini),
        "top_10_recall": float(top10_recall),
        "neg_rmse": float(-rmse),
        "spearman_corr": float(spearman),
    }


def ltv_candidate_evaluator(program_candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluator function conforming to AlphaEvolve API contract.
    Takes a candidate program dict, dynamically executes it, trains and evaluates the model,
    and returns evaluation scores and insights.
    """
    try:
        files = program_candidate.get("content", {}).get("files", [])
        if not files:
            raise ValueError("Candidate program has no files.")

        code = files[0].get("content", "")
        if not code.strip():
            raise ValueError("Candidate program file content is empty.")

        # Safe execution namespace with common analytical packages
        exec_namespace: Dict[str, Any] = {
            "__file__": str(Path.cwd() / "src" / "program.py"),
            "np": np,
            "numpy": np,
            "pd": pd,
            "pandas": pd,
            "Any": Any,
            "Mapping": Mapping,
            "__builtins__": __builtins__,
        }

        # Execute candidate code
        exec(code, exec_namespace)

        # Retrieve evaluate or engineer_features
        eval_func = exec_namespace.get("evaluate")
        engineer_func = exec_namespace.get("engineer_features")

        if callable(eval_func):
            metrics = eval_func({})
        elif callable(engineer_func):
            metrics = evaluate_feature_function(engineer_func)
        else:
            raise AttributeError("Candidate does not define 'evaluate' or 'engineer_features'.")

        primary_score = float(metrics.get("normalized_gini", -1e12))
        top10_score = float(metrics.get("top_10_recall", 0.0))
        neg_rmse = float(metrics.get("neg_rmse", -1e9))
        spearman = float(metrics.get("spearman_corr", 0.0))

        return {
            "scores": {
                "scores": [
                    {"metric": "normalized_gini", "score": primary_score},
                    {"metric": "top_10_recall", "score": top10_score},
                    {"metric": "neg_rmse", "score": neg_rmse},
                    {"metric": "spearman_corr", "score": spearman},
                ]
            },
            "insights": {
                "insights": [
                    {
                        "label": "Evaluation_Summary",
                        "text": (
                            f"Features evaluated successfully: Normalized Gini={primary_score:.4f}, "
                            f"Top-10% Recall={top10_score:.2f}%, neg_RMSE={neg_rmse:.2f}"
                        ),
                    }
                ]
            },
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        tb_summary = traceback.format_exc().splitlines()[-3:]
        full_detail = f"{error_msg} | {' '.join(tb_summary)}"
        logger.warning(f"Candidate evaluation failed: {full_detail}")

        return {
            "scores": {
                "scores": [
                    {"metric": "normalized_gini", "score": -1e12},
                    {"metric": "top_10_recall", "score": -1e12},
                    {"metric": "neg_rmse", "score": -1e12},
                    {"metric": "spearman_corr", "score": -1e12},
                ]
            },
            "insights": {
                "insights": [
                    {
                        "label": "Execution_Error",
                        "text": (
                            f"Execution failed with {error_msg}. "
                            "Ensure engineer_features(df: pd.DataFrame) -> pd.DataFrame preserves existing columns "
                            "and does not perform illegal division or invalid lookups."
                        ),
                    }
                ]
            },
        }


def load_initial_program_code(program_path: Optional[str | Path] = None) -> str:
    """Loads seed program code from src/program.py."""
    if program_path is None:
        program_path = Path(__file__).resolve().parent / "program.py"
    with open(program_path, "r", encoding="utf-8") as f:
        return f.read()
