"""
Main controller entrypoint for AlphaEvolve Mobile Game LTV Feature Engineering Evolution.

Usage:
    python -m src.run_evolution [--config config.yaml] [--programs N]

Key Workflow:
1. Load unified settings from config.yaml.
2. Initialize AlphaEvolveClient and AlphaEvolveExperiment.
3. Validate and split 3-way dataset (60% Train, 20% Eval, 20% Holdout) if needed.
4. Establish seed baseline metrics on the evaluation set.
5. Create and start AlphaEvolve experiment with Gemini Enterprise backend.
6. Run asynchronous controller loop with parallel candidate evaluation.
7. Retrieve top candidate (Champion), save artifacts, and evaluate on Holdout set.
"""

import argparse
import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import nest_asyncio
import numpy as np
import pandas as pd

from src.config import ProjectConfig, load_yaml_config, normalize_metric_name
from src.dataset import engineer_features as seed_engineer_features
from src.evaluate import (
    evaluate_feature_function,
    get_cached_splits,
    load_initial_program_code,
    ltv_candidate_evaluator,
)

# Apply nest_asyncio for nested asyncio loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_evolution")


def _extract_metric_score(program: Dict[str, Any], metric_name: str) -> float:
    """Safely extracts a score for metric_name from a program dict."""
    try:
        scores = program.get("evaluation", {}).get("scores", {}).get("scores", [])
        for s in scores:
            if s.get("metric") == metric_name:
                score = s.get("score")
                if score is not None:
                    return float(score)
    except Exception:
        pass
    return -float("inf")


def _save_json(data: Any, filepath: Path) -> None:
    """Helper to save formatted JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


async def main_async(args: argparse.Namespace) -> None:
    start_time = datetime.now()
    t0 = time.time()

    # 1. Load config
    raw_cfg = load_yaml_config(args.config)
    proj_cfg = ProjectConfig()

    gcp_cfg = raw_cfg.get("gcp", {})
    evo_cfg = raw_cfg.get("evolution", {})
    models_cfg = raw_cfg.get("models", [])

    # Task ID assignment & directory isolation
    task_id = args.task_id or datetime.now().strftime("task_%Y%m%d_%H%M%S")
    proj_cfg.task_id = task_id
    base_art_dir = proj_cfg.artifact_dir
    base_art_dir.mkdir(parents=True, exist_ok=True)
    task_art_dir = base_art_dir / task_id
    task_art_dir.mkdir(parents=True, exist_ok=True)

    # Maintain latest pointer for convenience
    try:
        latest_link = base_art_dir / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(task_id, target_is_directory=True)
    except Exception:
        pass
    try:
        (base_art_dir / "latest_task.txt").write_text(task_id, encoding="utf-8")
    except Exception:
        pass

    # Metric normalization
    raw_metric = args.metric or evo_cfg.get("primary_metric", "normalized_gini")
    primary_metric = normalize_metric_name(raw_metric)
    proj_cfg.primary_metric = primary_metric

    # Allow CLI overrides
    max_gen = int(args.programs) if args.programs is not None else int(evo_cfg.get("max_programs_generated", 20))
    max_eval = int(args.programs) if args.programs is not None else int(evo_cfg.get("max_programs_evaluated", 20))
    concurrency = int(args.concurrency) if args.concurrency is not None else int(evo_cfg.get("concurrency", 4))
    worker_concurrency = int(args.worker_concurrency) if args.worker_concurrency is not None else int(evo_cfg.get("worker_concurrency", 4))
    parallel_eval = bool(evo_cfg.get("parallel_evaluation", True))
    idle_timeout = int(evo_cfg.get("idle_timeout_s", 120))

    logger.info("=" * 70)
    logger.info(f"Starting AlphaEvolve LTV Feature Engineering Evolution (Task: {task_id})")
    logger.info(f"Project ID:         {gcp_cfg.get('project_id')}")
    logger.info(f"Engine / App ID:    {gcp_cfg.get('ge_app_id')}")
    logger.info(f"Target Programs:    max_generated={max_gen}, max_evaluated={max_eval}")
    logger.info(f"Concurrency:        cloud={concurrency}, local_workers={worker_concurrency}, parallel={parallel_eval}")
    logger.info(f"Primary Metric:     {primary_metric} (normalized from '{raw_metric}')")
    logger.info(f"Task Artifacts Dir: {task_art_dir}")
    logger.info("=" * 70)

    # 2. Dynamic import of alpha_evolve
    try:
        from alpha_evolve.client import AlphaEvolveClient
        from alpha_evolve.controller import run_controller_loop
        from alpha_evolve.experiment import AlphaEvolveExperiment
    except ImportError as e:
        logger.error(f"Failed to import alpha_evolve: {e}. Check gcp.alpha_evolve_path in config.yaml.")
        raise

    # 3. Ensure datasets are ready & cached
    logger.info("Preparing and validating 60%/20%/20% dataset splits...")
    train_df, eval_df, holdout_df = get_cached_splits(proj_cfg)
    logger.info(
        f"Datasets ready: Train={len(train_df):,} rows, "
        f"Eval={len(eval_df):,} rows, Holdout={len(holdout_df):,} rows."
    )

    # 4. Initialize AlphaEvolve Client & Experiment
    client = AlphaEvolveClient(
        project_id=gcp_cfg.get("project_id", ""),
        location=gcp_cfg.get("location", "global"),
        collection=gcp_cfg.get("collection", "default_collection"),
        engine=gcp_cfg.get("ge_app_id", ""),
        assistant=gcp_cfg.get("assistant", "default_assistant"),
        base_url=gcp_cfg.get("base_url", "discoveryengine.googleapis.com"),
    )

    experiment = AlphaEvolveExperiment(
        ae_client=client,
        evaluator_function=ltv_candidate_evaluator,
        max_programs_evaluated=max_eval,
        parallel_evaluation=parallel_eval,
    )

    # 5. Formulate experiment config
    exp_config = {
        "title": evo_cfg.get("title", "Mobile Game LTV Feature Engineering Evolution"),
        "problem_description": (
            f"{evo_cfg.get('problem_description', 'Evolve feature engineering transformations for 180-day mobile game LTV forecasting.')} "
            f"Specifically optimize {primary_metric} using a fixed LightGBM Tweedie regression baseline."
        ),
        "program_language": evo_cfg.get("program_language", "python"),
        "run_settings": {
            "max_programs": max_gen,
            "concurrency": concurrency,
        },
        "generation_settings": {
            "models": models_cfg,
            "context": (
                "Optimize engineer_features(df: pd.DataFrame) -> pd.DataFrame for mobile game LTV forecasting. "
                "Keep existing features intact, and engineer domain-specific interaction features: "
                "e.g. payer momentum, session event velocity, non-linear engagement ratios, and recency decay. "
                "Ensure all transformations are leak-free and robust to zero division."
            ),
        },
    }

    logger.info("Creating Gemini Enterprise session and AlphaEvolve experiment...")
    experiment.create_experiment(exp_config)

    # 6. Prepare Seed Program
    seed_code = load_initial_program_code()
    seed_path = task_art_dir / "seed_program.py"
    seed_path.write_text(seed_code, encoding="utf-8")
    logger.info(f"Saved initial seed program to {seed_path}")

    # Evaluate seed locally on evaluation set to establish baseline
    logger.info("Evaluating seed program baseline on Eval set...")
    seed_eval_metrics = evaluate_feature_function(seed_engineer_features, proj_cfg, eval_target="eval")
    logger.info(
        f"Seed Baseline (Eval Set): "
        f"Gini={seed_eval_metrics['normalized_gini']:.4f}, "
        f"Top-10% Recall={seed_eval_metrics['top_10_recall']:.2f}%, "
        f"RMSE={-seed_eval_metrics['neg_rmse']:.2f}, "
        f"Spearman={seed_eval_metrics['spearman_corr']:.4f}"
    )

    initial_program = {
        "content": {
            "files": [
                {
                    "path": "program.py",
                    "content": seed_code,
                }
            ]
        },
        "evaluation": {
            "scores": {
                "scores": [
                    {"metric": "normalized_gini", "score": float(seed_eval_metrics["normalized_gini"])},
                    {"metric": "top_10_recall", "score": float(seed_eval_metrics["top_10_recall"])},
                    {"metric": "neg_rmse", "score": float(seed_eval_metrics["neg_rmse"])},
                    {"metric": "spearman_corr", "score": float(seed_eval_metrics["spearman_corr"])},
                ]
            }
        },
    }

    experiment.create_initial_program(initial_program)
    experiment.start_experiment()

    # 7. Start Controller Loop
    logger.info("Starting controller loop for candidate generation and evaluation...")
    await run_controller_loop(
        experiment,
        num_samplers=concurrency,
        num_evaluators=worker_concurrency,
        idle_timeout_s=idle_timeout,
    )
    logger.info("Controller loop completed.")

    # 8. Retrieve All Evaluated Programs
    logger.info("Retrieving all evolved candidate programs from AlphaEvolve service...")
    res = experiment.list_programs() or {}
    if isinstance(res, dict):
        programs_list = res.get("alphaEvolvePrograms", [])
    elif isinstance(res, list):
        programs_list = res
    else:
        programs_list = []
    logger.info(f"Retrieved {len(programs_list)} candidate programs.")

    # Save evolution history
    history_path = task_art_dir / "evolution_history.json"
    _save_json(res if isinstance(res, dict) else {"alphaEvolvePrograms": programs_list}, history_path)
    logger.info(f"Saved full evolution history to {history_path}")

    # Rank programs by primary metric
    valid_programs = []
    for p in programs_list:
        score = _extract_metric_score(p, primary_metric)
        if score > -1e10:
            valid_programs.append((score, p))

    valid_programs.sort(key=lambda x: x[0], reverse=True)

    if not valid_programs:
        logger.warning("No valid candidate program found. Falling back to seed program as champion.")
        champion_program = initial_program
        champion_score = seed_eval_metrics.get(primary_metric, seed_eval_metrics["normalized_gini"])
        champion_code = seed_code
    else:
        champion_score, champion_program = valid_programs[0]
        champion_code = champion_program["content"]["files"][0]["content"]

    # Save champion code
    champ_path = task_art_dir / "champion_program.py"
    champ_path.write_text(champion_code, encoding="utf-8")
    logger.info(f"Champion program saved to {champ_path} (Eval Score: {champion_score:.4f})")

    # 9. Evaluate Seed vs Champion on 20% HOLDOUT SET
    logger.info("=" * 70)
    logger.info("Evaluating Seed vs Champion on the untouched 20% HOLDOUT SET...")
    logger.info("=" * 70)

    # Dynamic exec champion code to get engineer_features
    champ_ns: Dict[str, Any] = {
        "__file__": str(Path.cwd() / "src" / "program.py"),
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd,
        "__builtins__": __builtins__,
    }
    exec(champion_code, champ_ns)
    champ_engineer_func = champ_ns.get("engineer_features") or seed_engineer_features

    seed_holdout_metrics = evaluate_feature_function(seed_engineer_features, proj_cfg, eval_target="holdout")
    champ_holdout_metrics = evaluate_feature_function(champ_engineer_func, proj_cfg, eval_target="holdout")

    # Compute comparative delta
    gini_diff = champ_holdout_metrics["normalized_gini"] - seed_holdout_metrics["normalized_gini"]
    recall_diff = champ_holdout_metrics["top_10_recall"] - seed_holdout_metrics["top_10_recall"]
    seed_rmse = -seed_holdout_metrics["neg_rmse"]
    champ_rmse = -champ_holdout_metrics["neg_rmse"]
    rmse_diff = champ_rmse - seed_rmse
    rmse_pct = (rmse_diff / seed_rmse) * 100.0

    # Query GCP API statistics for telemetry
    exp_info = client.get_alpha_evolve_experiment(experiment.experiment_name) or {}
    api_stats = exp_info.get("stats", {})

    billed_in_tok = int(api_stats.get("inputTokenCount", 0)) if api_stats.get("inputTokenCount") else 0
    billed_out_tok = int(api_stats.get("outputTokenCount", 0)) if api_stats.get("outputTokenCount") else 0
    candidates_count = int(api_stats.get("candidatesCount", len(programs_list)))
    evaluated_count = int(api_stats.get("evaluatedCandidatesCount", experiment.stats.get("num_programs_evaluated", 0)))

    end_time = datetime.now()
    total_duration_s = time.time() - t0
    mins = int(total_duration_s // 60)
    secs = int(total_duration_s % 60)

    total_weight = sum(float(m.get("weight", 0.0)) for m in models_cfg) or 1.0
    model_calls = {}
    for m in models_cfg:
        m_name = m.get("name", "gemini-model")
        m_weight = float(m.get("weight", 0.0))
        est_calls = int(round(candidates_count * (m_weight / total_weight))) if candidates_count > 0 else 0
        model_calls[m_name] = {
            "weight": m_weight,
            "percentage": f"{(m_weight / total_weight) * 100:.0f}%",
            "calls": est_calls,
        }

    telemetry_data = {
        "task_id": task_id,
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(total_duration_s, 2),
        "duration_formatted": f"{mins:02d}m {secs:02d}s",
        "total_candidates_generated": candidates_count,
        "total_programs_evaluated": evaluated_count,
        "total_model_calls": candidates_count,
        "model_calls_by_model": model_calls,
        "token_usage": {
            "billed_input_tokens": billed_in_tok,
            "billed_output_tokens": billed_out_tok,
            "total_tokens": billed_in_tok + billed_out_tok,
        },
    }

    seed_pri_score = seed_eval_metrics.get(primary_metric, seed_eval_metrics["normalized_gini"])
    summary_metrics = {
        "task_id": task_id,
        "experiment_name": experiment.experiment_name,
        "total_programs_evaluated": evaluated_count,
        "primary_metric": primary_metric,
        "eval_set": {
            "seed_primary_score": float(seed_pri_score),
            "champion_primary_score": float(champion_score),
            "eval_score_gain": float(champion_score - seed_pri_score),
        },
        "holdout_set": {
            "seed": {
                "normalized_gini": float(seed_holdout_metrics["normalized_gini"]),
                "top_10_recall": float(seed_holdout_metrics["top_10_recall"]),
                "rmse": float(seed_rmse),
                "spearman_corr": float(seed_holdout_metrics["spearman_corr"]),
            },
            "champion": {
                "normalized_gini": float(champ_holdout_metrics["normalized_gini"]),
                "top_10_recall": float(champ_holdout_metrics["top_10_recall"]),
                "rmse": float(champ_rmse),
                "spearman_corr": float(champ_holdout_metrics["spearman_corr"]),
            },
            "improvement": {
                "delta_normalized_gini": float(gini_diff),
                "delta_top_10_recall": float(recall_diff),
                "delta_rmse": float(rmse_diff),
                "rmse_reduction_percent": float(-rmse_pct),
            },
        },
        "telemetry": telemetry_data,
    }

    metrics_path = task_art_dir / "champion_metrics.json"
    _save_json(summary_metrics, metrics_path)
    logger.info(f"Saved holdout comparison metrics and telemetry to {metrics_path}")

    # Log terminal summary table
    print("\n" + "=" * 70)
    print(" " * 20 + f"FINAL HOLDOUT BENCHMARK SUMMARY (Task: {task_id})")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Seed Baseline':<16} | {'Champion Code':<16} | {'Gain / Delta':<14}")
    print("-" * 70)
    print(
        f"{'Normalized Gini':<25} | {seed_holdout_metrics['normalized_gini']:<16.4f} | "
        f"{champ_holdout_metrics['normalized_gini']:<16.4f} | {gini_diff:+14.4f}"
    )
    print(
        f"{'Top-10% Revenue Recall':<25} | {seed_holdout_metrics['top_10_recall']:<15.2f}% | "
        f"{champ_holdout_metrics['top_10_recall']:<15.2f}% | {recall_diff:+13.2f}%"
    )
    print(
        f"{'RMSE (Holdout)':<25} | {seed_rmse:<16.2f} | "
        f"{champ_rmse:<16.2f} | {rmse_diff:+14.2f} ({rmse_pct:+.2f}%)"
    )
    print(
        f"{'Spearman Correlation':<25} | {seed_holdout_metrics['spearman_corr']:<16.4f} | "
        f"{champ_holdout_metrics['spearman_corr']:<16.4f} | "
        f"{champ_holdout_metrics['spearman_corr'] - seed_holdout_metrics['spearman_corr']:+14.4f}"
    )
    print("=" * 70)

    # 10. Automatically generate HTML report at the end of evolution run
    from src.report import generate_html_report

    report_path = generate_html_report(
        config=proj_cfg,
        task_id=task_id,
        artifact_dir=task_art_dir,
    )

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Evolution task '{task_id}' completed successfully!")
    print(f"Artifacts Directory:   {task_art_dir.resolve()}")
    print(f"HTML Evolution Report: {report_path.resolve()}")
    print("-" * 70)
    print("To open this report in your default browser, run:")
    print("    make report")
    print(f"  or:")
    print(f"    make report task_id={task_id}")
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaEvolve LTV Feature Engineering Evolution")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to unified config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--task-id",
        "--task_id",
        type=str,
        default=None,
        help="Custom task ID (default: timestamp-based task_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--programs",
        type=int,
        default=None,
        help="Overrides max_programs_generated and max_programs_evaluated (e.g. for testing make run programs=2)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Overrides cloud generation concurrency",
    )
    parser.add_argument(
        "--worker_concurrency",
        type=int,
        default=None,
        help="Overrides local evaluation worker concurrency",
    )
    parser.add_argument(
        "--metric",
        "--primary_metric",
        dest="metric",
        type=str,
        default=None,
        help="Primary metric for selecting champion (normalized_gini, top_10_recall, neg_rmse, spearman_corr)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main_async(cli_args))
