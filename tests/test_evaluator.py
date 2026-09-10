"""
Unit tests for candidate evaluator harness and program execution.
"""

import pytest

from src.evaluate import load_initial_program_code, ltv_candidate_evaluator


def test_seed_program_code_loading():
    code = load_initial_program_code()
    assert "# EVOLVE-BLOCK-START" in code
    assert "# EVOLVE-BLOCK-END" in code
    assert "def engineer_features" in code
    assert "def evaluate" in code


def test_candidate_evaluator_syntax_error():
    broken_candidate = {
        "content": {
            "files": [
                {
                    "path": "program.py",
                    "content": "def engineer_features(df):\n    this is invalid syntax !!!",
                }
            ]
        }
    }
    result = ltv_candidate_evaluator(broken_candidate)
    assert "scores" in result
    scores = result["scores"]["scores"]
    for s in scores:
        assert s["score"] <= -1e10
    assert "insights" in result
    assert "Execution failed" in result["insights"]["insights"][0]["text"]


def test_evaluate_feature_function_seed():
    from src.dataset import engineer_features
    from src.evaluate import evaluate_feature_function
    metrics = evaluate_feature_function(engineer_features)
    assert 0.8 < metrics["normalized_gini"] < 1.0
    assert metrics["top_10_recall"] > 50.0
    assert metrics["neg_rmse"] < 0
    assert "spearman_corr" in metrics


def test_evaluator_all_metrics():
    seed_code = load_initial_program_code()
    candidate = {
        "content": {
            "files": [
                {
                    "path": "program.py",
                    "content": seed_code,
                }
            ]
        }
    }
    result = ltv_candidate_evaluator(candidate)
    scores = {s["metric"]: s["score"] for s in result["scores"]["scores"]}
    assert "normalized_gini" in scores
    assert "top_10_recall" in scores
    assert "neg_rmse" in scores
    assert "spearman_corr" in scores

