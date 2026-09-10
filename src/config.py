"""
Configuration module for LTV modeling pipeline.
Loads configuration from config.yaml (Single Source of Truth) and defines paths,
feature schemas, GCP credentials, and model hyperparameters for LightGBM Tweedie regression.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import yaml


def _find_config_file() -> Path:
    """Finds config.yaml starting from current directory upwards."""
    current = Path(__file__).resolve().parent
    for _ in range(4):
        candidate = current / "config.yaml"
        if candidate.exists():
            return candidate
        current = current.parent
    return Path("config.yaml").resolve()


def load_yaml_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Loads YAML configuration dictionary."""
    path = Path(config_path) if config_path else _find_config_file()
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_metric_name(name: Optional[str]) -> str:
    """
    Normalizes user-supplied metric names and aliases to canonical AlphaEvolve metric names:
    - 'gini', 'normalized_gini' -> 'normalized_gini'
    - 'recall', 'top_10', 'top_10_recall' -> 'top_10_recall'
    - 'rmse', 'neg_rmse' -> 'neg_rmse'
    - 'spearman', 'spearman_corr' -> 'spearman_corr'
    """
    if not name:
        return "normalized_gini"
    clean = str(name).strip().lower().replace("-", "_")
    mapping = {
        "gini": "normalized_gini",
        "normalized_gini": "normalized_gini",
        "norm_gini": "normalized_gini",
        "top_10": "top_10_recall",
        "top10": "top_10_recall",
        "recall": "top_10_recall",
        "top_10_recall": "top_10_recall",
        "top_10_revenue_recall": "top_10_recall",
        "rmse": "neg_rmse",
        "neg_rmse": "neg_rmse",
        "spearman": "spearman_corr",
        "spearman_corr": "spearman_corr",
        "correlation": "spearman_corr",
    }
    if clean in mapping:
        return mapping[clean]
    raise ValueError(
        f"Unsupported metric '{name}'. Supported metrics are: normalized_gini, top_10_recall, neg_rmse, spearman_corr "
        "(aliases: gini, recall, rmse, spearman)."
    )


def get_latest_task_id(artifact_base_dir: Optional[Path] = None) -> Optional[str]:
    """Finds the most recent task_id under the artifacts directory."""
    if artifact_base_dir is None:
        cfg = ProjectConfig()
        artifact_base_dir = cfg.artifact_dir

    # 1. Check latest_task.txt pointer
    latest_file = artifact_base_dir / "latest_task.txt"
    if latest_file.exists():
        tid = latest_file.read_text(encoding="utf-8").strip()
        if tid and (artifact_base_dir / tid).is_dir():
            return tid

    # 2. Check latest symlink
    latest_link = artifact_base_dir / "latest"
    if latest_link.exists() and latest_link.is_dir():
        try:
            return latest_link.resolve().name
        except Exception:
            pass

    # 3. Search for task_* subdirectories sorted reverse
    if artifact_base_dir.exists():
        task_dirs = sorted(
            [d.name for d in artifact_base_dir.glob("task_*") if d.is_dir()],
            reverse=True,
        )
        if task_dirs:
            return task_dirs[0]

    return None


@dataclass
class ProjectConfig:
    # Task identification & Metric
    task_id: Optional[str] = None
    primary_metric: str = "normalized_gini"

    # Base paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = field(default=None)
    model_dir: Path = field(default=None)
    report_dir: Path = field(default=None)
    artifact_dir: Path = field(default=None)

    # Data file paths
    raw_wide_path: Path = field(default=None)
    train_split_path: Path = field(default=None)
    eval_split_path: Path = field(default=None)
    holdout_split_path: Path = field(default=None)
    holdout_preds_path: Path = field(default=None)
    model_path: Path = field(default=None)

    # Splitting configuration
    train_ratio: float = 0.60
    eval_ratio: float = 0.20
    holdout_ratio: float = 0.20
    test_size: float = 0.20  # backward-compatibility for holdout
    random_state: int = 42
    split_method: str = "stratified"

    # Columns
    id_col: str = "user_id"
    target_col: str = "ltv_d8_d180"
    payer_col: str = "is_payer_d8_d180"
    drop_cols: List[str] = field(default_factory=lambda: ["user_id", "ltv_d8_d180", "is_payer_d8_d180"])
    cat_cols: List[str] = field(default_factory=lambda: ["platform", "country_tier", "channel_tier"])

    # Raw config dict
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Load from config.yaml if present
        yaml_cfg = load_yaml_config()
        self.raw_config = yaml_cfg

        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.model_dir is None:
            self.model_dir = self.base_dir / "models"
        if self.report_dir is None:
            self.report_dir = self.base_dir / "reports"
        if self.artifact_dir is None:
            art_cfg = yaml_cfg.get("artifacts", {}).get("dir", "./artifacts")
            self.artifact_dir = (self.base_dir / art_cfg).resolve()

        ds_cfg = yaml_cfg.get("dataset", {})
        if self.raw_wide_path is None:
            raw_path = ds_cfg.get("raw_wide_path", "./data/train_wide.csv")
            self.raw_wide_path = (self.base_dir / raw_path).resolve()
        if self.train_split_path is None:
            tr_path = ds_cfg.get("train_split_path", "./data/train_split.csv")
            self.train_split_path = (self.base_dir / tr_path).resolve()
        if self.eval_split_path is None:
            ev_path = ds_cfg.get("eval_split_path", "./data/eval_split.csv")
            self.eval_split_path = (self.base_dir / ev_path).resolve()
        if self.holdout_split_path is None:
            ho_path = ds_cfg.get("holdout_split_path", "./data/holdout_split.csv")
            self.holdout_split_path = (self.base_dir / ho_path).resolve()
        if self.holdout_preds_path is None:
            self.holdout_preds_path = self.data_dir / "holdout_predictions.csv"
        if self.model_path is None:
            self.model_path = self.model_dir / "lgbm_tweedie.pkl"

        self.train_ratio = float(ds_cfg.get("train_ratio", self.train_ratio))
        self.eval_ratio = float(ds_cfg.get("eval_ratio", self.eval_ratio))
        self.holdout_ratio = float(ds_cfg.get("holdout_ratio", self.holdout_ratio))
        self.test_size = self.holdout_ratio
        self.random_state = int(ds_cfg.get("random_state", self.random_state))
        self.split_method = str(ds_cfg.get("split_method", self.split_method))

        # Ensure output directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        if self.task_id:
            self.task_dir.mkdir(parents=True, exist_ok=True)

        # Handle alpha_evolve import path configuration
        self._setup_alpha_evolve_path()

    @property
    def gcp_project_id(self) -> str:
        """Returns the GCP project ID configured in config.yaml."""
        return self.raw_config.get("gcp", {}).get("project_id", "")

    @property
    def task_dir(self) -> Path:
        """Returns the specific task directory if task_id is set, else artifact_dir."""
        if self.task_id:
            d = self.artifact_dir / self.task_id
            d.mkdir(parents=True, exist_ok=True)
            return d
        return self.artifact_dir

    def resolve_task_dir(self, task_id: Optional[str] = None) -> Path:
        """Resolves task directory: explicit task_id -> latest task_id -> artifact_dir."""
        tid = task_id or self.task_id or get_latest_task_id(self.artifact_dir)
        if tid:
            d = self.artifact_dir / tid
            if d.is_dir():
                return d
        return self.artifact_dir

    def _setup_alpha_evolve_path(self):
        """Resolves alpha_evolve directory from config.yaml and adds it to sys.path."""
        ae_path = self.raw_config.get("gcp", {}).get("alpha_evolve_path", "./alpha_evolve")
        resolved = (self.base_dir / ae_path).resolve()
        # If the path points to alpha_evolve directory, its parent needs to be in sys.path
        # so that `import alpha_evolve` or `from alpha_evolve import ...` works
        if resolved.is_dir():
            parent_dir = str(resolved.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            resolved_dir = str(resolved)
            if resolved_dir not in sys.path:
                sys.path.insert(0, resolved_dir)


# Default LightGBM Tweedie hyperparameters (frozen model baseline)
DEFAULT_TWEEDIE_PARAMS: Dict = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": 4,
    "verbose": -1,
}
