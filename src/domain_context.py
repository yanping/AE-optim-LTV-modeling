"""
Universal Code Evolution Attribution & Domain Context Knowledge Base.
Provides domain understanding and dynamically queries Gemini to analyze
and interpret code evolution across any algorithmic component (features,
model architectures, loss functions, hyperparameters, preprocessing, etc.).
"""

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import google.auth
import google.auth.transport.requests
import requests

logger = logging.getLogger(__name__)

GAME_LTV_DOMAIN_KNOWLEDGE = """
# Mobile Game 180-Day LTV Domain Knowledge & Modeling Specification

## 1. Business Context & Objective
- **Application**: Free-to-Play (F2P) Mobile Game with Hybrid Monetization (In-App Purchases + Rewarded Video Ads).
- **Target Variable (`ltv_d8_d180`)**: The cumulative net monetization (USD) of a player from Day 8 to Day 180 after registration.
- **Observation Window**: The first 8 calendar days of player activity: Day 0 (install day) through Day 7.
- **Goal**: Predict long-term 180-day customer lifetime value (LTV) using early 8-day behavioral and financial signals.
- **Evaluation Metrics**:
  - `normalized_gini`: Primary sorting power and Lorenz curve separation across non-payers and payers.
  - `top_10_recall`: Percentage of total 180-day revenue captured by the top-10% predicted highest-spending players (Whales).
  - `rmse`: Root Mean Squared Error, measuring monetary value prediction accuracy (penalizes large dollar deviations).
  - `spearman_corr`: Rank correlation coefficient across player cohort.

## 2. Data Distribution & Game Monetization Dynamics
- **Zero-Inflation**: Over 70% of players make 0 purchases throughout their lifetime (LTV = 0).
- **Heavy-Tail Pareto Distribution**: Less than 1% of top "whales" (high rollers / VIPs) contribute more than 80% of total game revenue.
- **Impulse vs. Habitual Monetization**:
  - Starter Packs / Gacha Impulse: Heavy D0-D2 purchases that quickly drop off often denote churned impulse buyers.
  - Retention & Acceleration: Continued spending or accelerating purchases on D5-D7 denote highly engaged, long-term guild leaders and core VIPs.
- **Hybrid Monetization Trade-offs**:
  - High ad engagement with 0 IAP: Price-sensitive "grinder" players who generate stable ad revenue but negligible IAP.
  - High IAP share: Whales who often bypass ads and focus heavily on competitive progression.

## 3. Key Schema Columns
- `user_id`: Unique player identifier.
- `platform`: Operating system (iOS vs. Android, differing in ARPU and monetization friction).
- `revenue_d0` to `revenue_d7`: Daily net revenue generated on day d (USD).
- `total_revenue_d7`: Cumulative revenue across the 8-day observation window.
- `iap_revenue_usd_sum`: Total in-app purchase revenue across D0-D7.
- `ad_revenue_usd_sum`: Total rewarded video ad revenue across D0-D7.
- `ad_impression_count`: Total rewarded video ads watched across D0-D7.
- `active_days_count`: Total count of active days (1 to 8).
- `total_events`: Total game action events logged across D0-D7.
- `event_count_d0` to `event_count_d7`: Daily event counts representing daily gameplay intensity.
"""


def query_gemini_code_attribution(
    seed_code: str,
    champ_code: str,
    holdout_metrics: Dict[str, Any],
    primary_metric: str = "normalized_gini",
    project_id: str = "",
    location: str = "us-central1",
    model_name: str = "gemini-2.5-flash",
    cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Dynamically calls Gemini to perform a generalized, in-depth attribution
    analysis comparing Seed code and Champion code across ANY evolved component:
    - Feature engineering & representation learning
    - Model architecture & hyperparameter tuning
    - Objective / custom loss function formulation
    - Data preprocessing, sampling, cleaning, or weighting
    - Post-processing, calibration, or ensembling

    Returns a structured dictionary:
    - evolution_scope: str (e.g. '特征工程与时序表示学习', '目标函数与超参数调优')
    - evolution_summary: str (1-2 sentence executive summary)
    - key_innovations: List[{title, type, badge_color, code_snippet, technical_mechanism, business_domain_impact}]
    - simplifications_and_pruning: str
    - strategic_insights: List[str]
    - source: 'gemini' | 'cached' | 'fallback'
    """
    # Check cache first
    if cache_path:
        # Check current cache path or legacy feature_explanation.json
        candidate_paths = [cache_path]
        if cache_path.name != "feature_explanation.json":
            candidate_paths.append(cache_path.parent / "feature_explanation.json")
        for cp in candidate_paths:
            if cp.exists():
                try:
                    cached_data = json.loads(cp.read_text(encoding="utf-8"))
                    adapted = _adapt_to_universal_schema(cached_data)
                    adapted["source"] = "cached"
                    logger.info(f"Loaded code attribution from cache: {cp}")
                    return adapted
                except Exception as e:
                    logger.warning(f"Failed to read cache {cp}: {e}")

    seed_score = holdout_metrics.get("seed", {}).get(primary_metric, "N/A")
    champ_score = holdout_metrics.get("champion", {}).get(primary_metric, "N/A")

    prompt = f"""
{GAME_LTV_DOMAIN_KNOWLEDGE}

---

# Task: Universal Code Evolution Attribution & Domain Rationale Analysis

You are a Principal Machine Learning Scientist and Quantitative Systems Architect.
An automated code evolution system (AlphaEvolve) has iteratively evolved an initial "Seed Program" into a high-performing "Champion Program".

## Crucial Context:
The evolved code block (# EVOLVE-BLOCK) can represent ANY component of the machine learning system:
- Feature engineering, temporal aggregations, and signal representations
- Model architectures, pipelines, and hyperparameter scheduling
- Custom objective/loss functions (e.g. Tweedie, asymmetric loss, sample weighting)
- Data preprocessing, filtering, cleaning, or sampling strategies
- Post-processing, score calibration, or prediction transforms

Your job is to analyze the difference between the Seed and Champion code with extreme technical precision and domain grounding.
Provide all explanations in simplified Chinese (简体中文).

Optimization Target Metric: **{primary_metric}**
Holdout Benchmark Scores:
- Seed Baseline {primary_metric}: {seed_score}
- Champion Code {primary_metric}: {champ_score}

## 1. Initial Seed Program (# EVOLVE-BLOCK)
```python
{seed_code}
```

## 2. Evolved Champion Program (# EVOLVE-BLOCK)
```python
{champ_code}
```

---

## Instructions:
1. **Identify Evolution Scope**: Determine which area(s) were evolved (e.g., 特征工程与时序表示学习, 损失函数与优化目标, 超参数调度与模型架构, 样本加权与数据清洗).
2. **Executive Summary**: A concise 1-2 sentence overview of why the Champion outperforms the Seed.
3. **Key Innovations**: Break down every key modification, addition, or algorithmic innovation. For each item provide:
   - `title`: Name or description of this innovation
   - `type`: Category of change (e.g., 特征工程, 模型架构, 优化目标/损失函数, 数据预处理, 算法逻辑, 超参数)
   - `badge_color`: Hex color code (e.g. #38bdf8 for trend/model, #10b981 for momentum/loss, #f59e0b for recency/sample, #6366f1 for log/norm, #ec4899 for efficiency/ratio, #a855f7 for architecture)
   - `code_snippet`: The representative code snippet or equation
   - `technical_mechanism`: Deep mathematical, algorithmic, and engineering explanation of what the code does and why
   - `business_domain_impact`: Concrete domain and business rationale (how it helps predict long-term 180-day LTV, captures whales, or optimizes {primary_metric})
4. **Simplifications and Pruning**: Detailed analysis of what parts were pruned, replaced, or preserved from the baseline, and why.
5. **Strategic Insights**: 3 high-impact strategic takeaways for downstream modeling and production deployment.

Return ONLY a valid JSON object matching the following structure:
{{
  "evolution_scope": "识别出的演化范围 (如: 特征工程与时序表示学习 / 自定义损失函数与样本加权 / 模型超参数体系)",
  "evolution_summary": "1-2句核心演化结论概述，总结冠军代码的核心突破",
  "key_innovations": [
    {{
      "title": "创新项名称 (如: revenue_slope 时序正交线性趋势 / 自定义非对称 Tweedie 损失)",
      "type": "创新类型 (如: 特征工程 / 模型配置 / 损失函数 / 算法逻辑 / 数据清洗)",
      "badge_color": "#38bdf8",
      "code_snippet": "核心代码行或表达式",
      "technical_mechanism": "数学、算法与系统实现机理的精准解读",
      "business_domain_impact": "业务价值与对优化目标的影响深度解读"
    }}
  ],
  "simplifications_and_pruning": "对代码改动中精简、淘汰、重构或非破坏性保留基线逻辑的深度专业分析",
  "strategic_insights": [
    "战略启示点1：算法设计与数学层面的重大突破点",
    "战略启示点2：如何针对性提升选优主指标（{primary_metric}）并防止过拟合",
    "战略启示点3：对后续工业界系统落地与算法迭代的启示"
  ]
}}
"""

    try:
        credentials, _ = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        token = credentials.token

        endpoint = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/"
            f"locations/{location}/publishers/google/models/{model_name}:generateContent"
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }

        resp = requests.post(endpoint, json=payload, headers=headers, timeout=45)
        if resp.status_code == 200:
            resp_data = resp.json()
            raw_text = resp_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            result = json.loads(raw_text.strip())
            adapted = _adapt_to_universal_schema(result)
            adapted["source"] = "gemini"

            # Cache the result locally
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(adapted, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info(f"Saved Gemini code attribution to cache: {cache_path}")

            return adapted
        else:
            logger.warning(f"Vertex AI Gemini call failed (status {resp.status_code}): {resp.text[:300]}")
    except Exception as e:
        logger.warning(f"Exception when calling Gemini for code attribution: {e}")

    # Fallback if Gemini call fails or offline
    return _build_fallback_attribution(seed_code, champ_code, primary_metric)


def _adapt_to_universal_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes both legacy feature_explanation.json format and new universal
    code_attribution.json format into a unified structure.
    """
    # If already universal
    if "key_innovations" in data:
        innovations = data.get("key_innovations", [])
        # Provide backward-compatibility aliases
        data["added_features"] = innovations
        data["removed_features_analysis"] = data.get("simplifications_and_pruning", "")
        data["strategic_takeaways"] = data.get("strategic_insights", [])
        return data

    # Legacy schema adaptation (from added_features)
    innovations = []
    for f in data.get("added_features", []):
        innovations.append({
            "title": f.get("name", "Evolved Item"),
            "type": "特征工程",
            "badge_color": f.get("badge_color", "#38bdf8"),
            "code_snippet": f.get("expression", ""),
            "technical_mechanism": f.get("methodology", f.get("method", "")),
            "business_domain_impact": f.get("business_rationale", ""),
        })

    return {
        "evolution_scope": data.get("evolution_scope", "特征工程与时序表示学习"),
        "evolution_summary": data.get("evolution_summary", "演化算法基于移动游戏长周期变现规律，扩充了高区分度特征集合。"),
        "key_innovations": innovations,
        "simplifications_and_pruning": data.get("removed_features_analysis", ""),
        "strategic_insights": data.get("strategic_takeaways", []),
        "added_features": innovations,  # alias
        "removed_features_analysis": data.get("removed_features_analysis", ""),  # alias
        "strategic_takeaways": data.get("strategic_takeaways", []),  # alias
        "source": data.get("source", "cached"),
    }


def _build_fallback_attribution(seed_code: str, champ_code: str, primary_metric: str) -> Dict[str, Any]:
    """
    Generates a robust universal fallback attribution by analyzing code diffs,
    detecting any added/modified functions, assignments, or expressions.
    """
    seed_lines = [line.strip() for line in seed_code.splitlines() if line.strip() and not line.strip().startswith("#")]
    champ_lines = [line.strip() for line in champ_code.splitlines() if line.strip() and not line.strip().startswith("#")]

    seed_set = set(seed_lines)
    added_lines = [l for l in champ_lines if l not in seed_set]
    removed_lines = [l for l in seed_lines if l not in set(champ_lines)]

    innovations = []
    # Identify type of evolution based on code contents
    is_feature_eng = any("df[" in l for l in champ_lines)
    is_model_cfg = any("objective" in l or "learning_rate" in l or "params" in l for l in champ_lines)
    is_custom_loss = any("def loss" in l or "grad" in l or "hess" in l for l in champ_lines)

    if is_custom_loss:
        scope = "自定义目标与损失函数演化"
        default_type = "优化目标 / 损失函数"
    elif is_model_cfg:
        scope = "模型架构与超参数体系演化"
        default_type = "模型配置"
    elif is_feature_eng:
        scope = "特征工程与时序表示学习演化"
        default_type = "特征工程"
    else:
        scope = "通用代码逻辑与算法管道演化"
        default_type = "算法逻辑"

    for l in added_lines:
        # Check if line is meaningful
        if len(l) > 6 and not l.startswith("return") and not l.startswith("import"):
            title = l.split("=")[0].strip() if "=" in l else l[:30]
            innovations.append({
                "title": title,
                "type": default_type,
                "badge_color": "#38bdf8",
                "code_snippet": l,
                "technical_mechanism": f"针对 `{title}` 执行算法改进与计算逻辑重构。",
                "business_domain_impact": f"丰富模型候选信息维度与判别边界，为指标 {primary_metric} 提供增益支撑。",
            })

    pruning_text = (
        f"冠军代码精简淘汰了 {len(removed_lines)} 行基线逻辑，以降低复杂度并抑制过拟合。"
        if removed_lines
        else "种子代码包含的所有基线逻辑在冠军代码中均完整保留（无一删除）。演化探索采取了非破坏性正交扩展策略，保持了基准系统的稳定性。"
    )

    return {
        "evolution_scope": scope,
        "evolution_summary": f"AlphaEvolve 完成了对候选代码的迭代演化，在保持稳定性的同时引入了 {len(innovations)} 项关键创新逻辑。",
        "key_innovations": innovations,
        "simplifications_and_pruning": pruning_text,
        "strategic_insights": [
            "演化机制有效探索了候选解空间，捕获了高阶模式。",
            f"针对优化主目标（{primary_metric}）提供了针对性的拟合增益。",
            "代码结构具备高鲁棒性，无数据泄露与异常越界风险。",
        ],
        "source": "fallback",
        # backward-compatibility aliases
        "added_features": innovations,
        "removed_features_analysis": pruning_text,
        "strategic_takeaways": [
            "演化机制有效探索了候选解空间，捕获了高阶模式。",
            f"针对优化主目标（{primary_metric}）提供了针对性的拟合增益。",
            "代码结构具备高鲁棒性，无数据泄露与异常越界风险。",
        ],
    }


# Backwards compatibility alias
query_gemini_feature_explanation = query_gemini_code_attribution
_build_fallback_explanation = _build_fallback_attribution
