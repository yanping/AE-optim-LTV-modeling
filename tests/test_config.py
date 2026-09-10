"""
Unit tests for configuration loading and validation.
"""

from pathlib import Path
import pytest

from src.config import ProjectConfig, load_yaml_config


def test_config_yaml_loading():
    cfg = load_yaml_config("config.yaml")
    assert "gcp" in cfg
    assert "evolution" in cfg
    assert "dataset" in cfg

    assert "project_id" in cfg["gcp"]
    assert "ge_app_id" in cfg["gcp"]
    assert len(cfg["gcp"]["project_id"]) > 0
    assert len(cfg["gcp"]["ge_app_id"]) > 0
    assert cfg["evolution"]["primary_metric"] == "normalized_gini"


def test_project_config_initialization():
    proj = ProjectConfig()
    assert proj.train_ratio == 0.60
    assert proj.eval_ratio == 0.20
    assert proj.holdout_ratio == 0.20
    assert proj.target_col == "ltv_d8_d180"
    assert proj.raw_wide_path.name == "train_wide.csv"
    assert proj.primary_metric == "normalized_gini"


def test_normalize_metric_name():
    from src.config import normalize_metric_name

    assert normalize_metric_name("gini") == "normalized_gini"
    assert normalize_metric_name("normalized_gini") == "normalized_gini"
    assert normalize_metric_name("top_10_recall") == "top_10_recall"
    assert normalize_metric_name("recall") == "top_10_recall"
    assert normalize_metric_name("rmse") == "neg_rmse"
    assert normalize_metric_name("neg_rmse") == "neg_rmse"
    assert normalize_metric_name("spearman") == "spearman_corr"
    assert normalize_metric_name("spearman_corr") == "spearman_corr"

    with pytest.raises(ValueError):
        normalize_metric_name("invalid_metric_xyz")


def test_task_dir_resolution(tmp_path):
    from src.config import get_latest_task_id

    # Test empty dir
    assert get_latest_task_id(tmp_path) is None

    # Test with task folders
    (tmp_path / "task_20260909_100000").mkdir()
    (tmp_path / "task_20260909_120000").mkdir()
    assert get_latest_task_id(tmp_path) == "task_20260909_120000"

    # Test with latest_task.txt pointer
    (tmp_path / "latest_task.txt").write_text("task_20260909_100000", encoding="utf-8")
    assert get_latest_task_id(tmp_path) == "task_20260909_100000"

    # Test ProjectConfig task_dir
    cfg = ProjectConfig(artifact_dir=tmp_path, task_id="task_custom_1")
    assert cfg.task_dir == tmp_path / "task_custom_1"

