"""
LightGBM Tweedie Model implementation for Mobile Game LTV Forecasting.
Compound Poisson-Gamma Tweedie regression natively handles zero-inflated continuous positive data.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


class BaseLTVModel(ABC):
    """Abstract Base Class for LTV models."""

    def __init__(self, name: str):
        self.name = name
        self.feature_names_: List[str] = []

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseLTVModel":
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        pass

    def save(self, path: Union[str, Path]):
        """Serializes model to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved model '{self.name}' to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseLTVModel":
        """Deserializes model from disk."""
        return joblib.load(path)


class LightGBMTweedieModel(BaseLTVModel):
    """
    LightGBM model with Tweedie distribution objective (Compound Poisson-Gamma).
    Optimized for zero-inflated continuous positive distributions.
    """

    def __init__(
        self,
        tweedie_variance_power: float = 1.5,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        n_estimators: int = 500,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = 4,
        early_stopping_rounds: int = 40,
    ):
        super().__init__(name="LightGBM_Tweedie")
        self.tweedie_variance_power = tweedie_variance_power
        self.params = {
            "objective": "tweedie",
            "tweedie_variance_power": tweedie_variance_power,
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": -1,
        }
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.booster_: Optional[lgb.LGBMRegressor] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LightGBMTweedieModel":
        self.feature_names_ = list(X_train.columns)
        self.booster_ = lgb.LGBMRegressor(**self.params, n_estimators=self.n_estimators)

        callbacks = []
        eval_X = None
        eval_y = None
        if X_val is not None and y_val is not None:
            eval_X = X_val
            eval_y = y_val
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self.booster_.fit(
            X_train,
            y_train,
            eval_X=eval_X,
            eval_y=eval_y,
            callbacks=callbacks if callbacks else None,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.booster_ is not None, "Model is not fitted yet."
        preds = self.booster_.predict(X)
        return np.maximum(preds, 0.0)

    def get_feature_importance(self) -> pd.DataFrame:
        assert self.booster_ is not None, "Model is not fitted yet."
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.booster_.feature_importances_,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
