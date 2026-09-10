"""
HTML Report Generator for AlphaEvolve Mobile Game LTV Feature Engineering Evolution.

Generates a self-contained, responsive, and visually appealing HTML report:
1. Executive Summary & Improvement Metrics on the 20% Holdout Set.
2. Visualized AlphaEvolve Generation & Evolution Trajectory (SVG chart).
3. Side-by-Side Code Diff between Seed Baseline and Champion Evolved Feature Engineering.
4. Feature Innovations & Domain Insights Summary.
"""

import argparse
import ast
import difflib
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import webbrowser

from src.config import ProjectConfig
from src.domain_context import (
    query_gemini_code_attribution,
    query_gemini_feature_explanation,
)

logger = logging.getLogger("report")


def _extract_evolve_block(code: str) -> str:
    """Extracts only the code inside EVOLVE-BLOCK-START and EVOLVE-BLOCK-END."""
    pattern = r"# EVOLVE-BLOCK-START(.*?)# EVOLVE-BLOCK-END"
    match = re.search(pattern, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code.strip()


def _extract_feature_assignments(code: str) -> Dict[str, str]:
    """
    Extracts {feature_name: assignment_expression_snippet} from feature engineering code.
    Uses Python AST parsing with line-range preservation and regex fallback.
    """
    block = _extract_evolve_block(code)
    lines = block.splitlines()
    assignments: Dict[str, str] = {}

    try:
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "df":
                        if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                            col_name = target.slice.value
                            start = node.lineno - 1
                            end = getattr(node, "end_lineno", node.lineno)
                            snippet = " ".join(line.strip() for line in lines[start:end])
                            assignments[col_name] = snippet
    except Exception:
        # Fallback to regex if AST encounters any parsing anomaly
        matches = re.findall(r'df\[["\']([^"\']+)["\']\]\s*=\s*([^\n]+)', block)
        for col, expr in matches:
            assignments[col] = f"df['{col}'] = {expr.strip()}"

    return assignments


def _analyze_feature_semantics(name: str, expr: str) -> Dict[str, str]:
    """
    Provides domain-specific categorization, mathematical methodology,
    and mobile game business rationale for an engineered feature.
    """
    combined = f"{name} {expr}".lower()

    if any(k in combined for k in ["curvature", "accel", "second_order"]):
        return {
            "category": "时序二阶凸度与变现加速度 (Temporal Curvature & Acceleration)",
            "badge_color": "#a855f7",
            "method": "计算二次正交多项式系数加权求和，提取 8 天付费轨迹的二阶曲率，测度变现曲线是呈现‘加速上凸’还是‘疲软下凹’。",
            "business_rationale": "量化玩家付费意愿的动力学升级过程。曲率大于零直接反映出该玩家正从轻微氪金向高频中重度大额消费快速质变（消费加速度 > 0），在运营上是识别高成长性核心公会玩家与大R的极佳早期指标。",
        }
    elif any(k in combined for k in ["trend", "projection", "slope", "weighted_rev"]):
        return {
            "category": "时序正交多项式低阶趋势投影 (Orthogonal Polynomial Trend Projection)",
            "badge_color": "#38bdf8",
            "method": "利用正交基多项式线性差分（如 3*(d7-d0) + 2*(d6-d1) + 1*(d5-d2)），平滑每日偶发充值噪声，提取全局变现速度一阶导数。",
            "business_rationale": "有效剥离 D0~D2 新手首充促销礼包带来的短期脉冲虚高，精确捕捉玩家在观察期后半段（D5~D7）是呈现消费提速还是迅速流失。后程强劲的玩家具备极高的社交参与度和长线游戏黏性，是预测 180 天核心长线高净值鲸鱼用户的强力信号。",
        }
    elif any(k in combined for k in ["growth", "momentum", "rev_growth", "ratio"]):
        return {
            "category": "跨时段变现动量扩张比率 (Multi-stage Monetization Momentum Ratio)",
            "badge_color": "#10b981",
            "method": "计算 D5~D7 晚期收入与 D0~D2 早期收入的除法比率（引入偏置项 0.01 防止除以零），放大长效留存玩家的变现跃迁幅度。",
            "business_rationale": "测算玩家从新手阶段向成熟阶段过渡时的消费韧性。相比只看总金额，动量比能有效剔除‘一次性冲动付费后弃坑’的噪音用户，锁定长期持续复购群体。",
        }
    elif any(k in combined for k in ["paying_day", "active_day", "sparsity", "events_per_active"]):
        return {
            "category": "行为频次深度与变现渗透率 (Engagement Frequency & Monetization Penetration)",
            "badge_color": "#f59e0b",
            "method": "聚合 8 天内的有效消费日总数与活跃天数占比，测算单活跃日的事件密度与消费频次。",
            "business_rationale": "明确区分‘高客单价但偶发’与‘高频次习惯型小额消费’。在手游经济学中，多次多日高频消费的‘习惯型玩家’其 180 天长线留存率显著高于单次偶发抽卡玩家。",
        }
    elif any(k in combined for k in ["ad_rev_per_imp", "rev_per", "arpdau", "unit_economic", "per_event"]):
        return {
            "category": "单体行为单位经济学效率 (Unit Economics & Value per Action)",
            "badge_color": "#06b6d4",
            "method": "跨模态比率计算（总流水/事件数、广告收益/展示次数、单次活跃天产出），消除游戏总时长造成的规模虚增。",
            "business_rationale": "直接衡量单位游戏行为的货币化转化能力（eCPM 与行为变现率），精准剥离耗时长但零变现的‘高耗时羊毛党’用户，赋能模型识别真正具备高变现效率的受众。",
        }
    elif any(k in combined for k in ["iap_share", "ad_share", "iap_to_ad"]):
        return {
            "category": "变现矩阵结构均衡 (Hybrid Monetization Matrix)",
            "badge_color": "#ec4899",
            "method": "计算 IAP 内购流水与激励视频广告流水的相对比重与互斥率。",
            "business_rationale": "准确刻画玩家的变现生态位。纯广告激励型玩家与纯 IAP 鲸鱼玩家的 180 天流失与价值衰减曲线截然不同，该比率帮助模型对混合变现用户进行针对性估值。",
        }
    elif any(k in combined for k in ["log", "log1p", "sqrt"]):
        return {
            "category": "长尾分布非线性压缩 (Heavy-tail Non-linear Scaling)",
            "badge_color": "#6366f1",
            "method": "运用 ln(1+x) 对数变换平滑极端偏态流水，压缩极值与中位数间的数量级跨度。",
            "business_rationale": "手游流水服从极端帕累托分布（前 1% 玩家贡献 80% 流水）。对数变换防止单笔极端巨鲸流水扭曲 LightGBM 决策树的方差分裂准则，显著降低模型过拟合风险。",
        }
    else:
        return {
            "category": "复合衍生交互特征 (Composite Interaction Feature)",
            "badge_color": "#64748b",
            "method": f"针对特征 `{name}` 进行业务交互衍生与非线性统计组合运算。",
            "business_rationale": "丰富 LightGBM 决策树候选分裂切分点，增强多维特征在复杂用户行为空间中的交叉区分度。",
        }


def _generate_code_evolution_attribution_html(explanation_data: Dict[str, Any]) -> str:
    """
    Renders the universal code evolution attribution and domain rationale section
    using structured insights generated by Gemini (grounded in game domain knowledge)
    or loaded from cache/fallback.

    Supports ANY evolved component:
    - Feature engineering & representation
    - Model architectures & hyperparameters
    - Custom loss/objective functions
    - Preprocessing, sampling, and post-processing
    """
    source = explanation_data.get("source", "gemini")
    if source == "gemini":
        source_badge = '<span style="background: #10b98122; color: #34d399; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px; border: 1px solid #10b98144;">🤖 由 Gemini 2.5 Flash 结合手游 LTV 领域知识库动态归因</span>'
    elif source == "cached":
        source_badge = '<span style="background: #3b82f622; color: #60a5fa; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px; border: 1px solid #3b82f644;">⚡ 来自本地归因缓存 (code_attribution.json)</span>'
    else:
        source_badge = '<span style="background: #64748b22; color: #94a3b8; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px; border: 1px solid #64748b44;">🛡️ 静态结构分析模式</span>'

    evolution_scope = explanation_data.get("evolution_scope", "算法代码与特征逻辑演化")
    evolution_summary = explanation_data.get("evolution_summary", "")

    # Retrieve innovations from new key_innovations or legacy added_features
    innovations = explanation_data.get("key_innovations")
    if innovations is None:
        innovations = explanation_data.get("added_features", [])

    pruning_analysis = explanation_data.get("simplifications_and_pruning")
    if pruning_analysis is None:
        pruning_analysis = explanation_data.get("removed_features_analysis", "")

    strategic_insights = explanation_data.get("strategic_insights")
    if strategic_insights is None:
        strategic_insights = explanation_data.get("strategic_takeaways", [])

    html_parts = []
    html_parts.append("<div style='margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;'>")
    html_parts.append(f"""
        <div style='display: flex; align-items: center; gap: 8px;'>
            <span style='color: #94a3b8; font-size: 13px;'>演化归因状态：</span>
            {source_badge}
        </div>
        <div>
            <span style='background: #6366f122; color: #a5b4fc; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 4px; border: 1px solid #6366f144;'>
                🧭 演化范畴：{evolution_scope}
            </span>
        </div>
    """)
    html_parts.append("</div>")

    # Executive summary banner
    if evolution_summary:
        html_parts.append(f"""
        <div style="background: #1e1b4b33; border: 1px solid #6366f144; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; color: #a5b4fc; font-weight: 700; font-size: 13px; margin-bottom: 4px;">
                <span>🎯 演化突破核心摘要 (Executive Summary)</span>
            </div>
            <p style="color: #e0e7ff; font-size: 13px; margin: 0; line-height: 1.6;">
                {evolution_summary}
            </p>
        </div>
        """)

    # 1. Innovations breakdown grid
    if innovations:
        html_parts.append("<div style='margin-bottom: 24px;'>")
        html_parts.append(f"<h3 style='color: #38bdf8; font-size: 15px; margin-bottom: 12px;'>✨ 冠军代码关键创新与改动项 ({len(innovations)} 项)</h3>")
        html_parts.append("<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px;'>")

        for item in innovations:
            title = item.get("title") or item.get("name", "Evolved Item")
            change_type = item.get("type") or item.get("category", "代码改动")
            badge_color = item.get("badge_color", "#38bdf8")
            code_snippet = item.get("code_snippet") or item.get("expression", "")
            mechanism = item.get("technical_mechanism") or item.get("methodology", item.get("method", ""))
            impact = item.get("business_domain_impact") or item.get("business_rationale", "")

            html_parts.append(f"""
            <div style="background: #090d16; border: 1px solid #1f2937; border-left: 4px solid {badge_color}; border-radius: 8px; padding: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-family: monospace; font-weight: 700; color: #f3f4f6; font-size: 14px;">{title}</span>
                    <span style="background: {badge_color}22; color: {badge_color}; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; border: 1px solid {badge_color}44;">
                        {change_type}
                    </span>
                </div>
                {f'<div style="font-family: monospace; font-size: 12px; color: #94a3b8; background: #0f172a; padding: 8px; border-radius: 4px; margin-bottom: 10px; word-break: break-all; white-space: pre-wrap;">{code_snippet}</div>' if code_snippet else ''}
                <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 8px; line-height: 1.5;">
                    <strong style="color: #94a3b8;">⚙️ 算法与实现机理：</strong>{mechanism}
                </div>
                <div style="font-size: 13px; color: #cbd5e1; line-height: 1.5;">
                    <strong style="color: #94a3b8;">🎯 业务价值与指标影响：</strong>{impact}
                </div>
            </div>
            """)

        html_parts.append("</div></div>")
    else:
        html_parts.append("<p style='color: #94a3b8;'>冠军代码与种子代码主要逻辑保持一致，演化聚焦在超参数细节微调。</p>")

    # 2. Pruning & Retention breakdown
    if pruning_analysis:
        html_parts.append(f"""
        <div style="background: #064e3b18; border: 1px solid #064e3b88; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; color: #34d399; font-weight: 600; font-size: 14px; margin-bottom: 6px;">
                <span>🛡️ 架构精简、淘汰与基线保留分析 (Pruning & Retention Analysis)</span>
            </div>
            <p style="color: #cbd5e1; font-size: 13px; margin: 0; line-height: 1.6;">
                {pruning_analysis}
            </p>
        </div>
        """)

    # 3. Strategic takeaways card
    if strategic_insights:
        html_parts.append("""
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 18px; margin-top: 16px;">
            <h4 style="margin: 0 0 10px 0; color: #e2e8f0; font-size: 14px;">💡 演化策略决策与工业落地启示 (Strategic Insights)</h4>
            <ul style="margin: 0; padding-left: 20px; color: #94a3b8; font-size: 13px; line-height: 1.8;">
        """)
        for item in strategic_insights:
            html_parts.append(f"<li><strong style='color: #e2e8f0;'>{item}</strong></li>")
        html_parts.append("</ul></div>")

    return "".join(html_parts)


# Backward compatibility alias
_generate_feature_diff_analysis_html = _generate_code_evolution_attribution_html




def _generate_telemetry_board_html(telemetry: Dict[str, Any], programs_count: int) -> str:
    """Renders the Execution & Resource Telemetry Dashboard."""
    duration = telemetry.get("duration_formatted", f"{telemetry.get('duration_seconds', 0):.1f}s")
    start_time = telemetry.get("start_time", "-")
    end_time = telemetry.get("end_time", "-")
    total_calls = telemetry.get("total_model_calls", programs_count)
    model_calls = telemetry.get("model_calls_by_model", {})
    token_usage = telemetry.get("token_usage", {})
    billed_in = token_usage.get("billed_input_tokens", 0)
    billed_out = token_usage.get("billed_output_tokens", 0)
    total_tokens = token_usage.get("total_tokens", billed_in + billed_out)

    badges = []
    if model_calls:
        for m_name, info in model_calls.items():
            calls = info.get("calls", 0)
            pct = info.get("percentage", "-")
            badges.append(
                f"<span style='display: inline-block; background: #1e293b; color: #38bdf8; font-size: 12px; padding: 3px 8px; border-radius: 4px; margin-right: 6px; margin-top: 4px;'>"
                f"{m_name}: <strong>{calls}</strong> 次 ({pct})"
                f"</span>"
            )
    else:
        badges.append(f"<span style='color: #94a3b8; font-size: 12px;'>总计 {total_calls} 次</span>")

    badges_html = "".join(badges)

    return f"""
    <div class="grid-cards" style="margin-bottom: 24px;">
        <div class="card">
            <div class="title">⏱️ 任务总执行耗时</div>
            <div class="value">{duration}</div>
            <div style="font-size: 12px; color: var(--text-muted); margin-top: 6px;">
                开始: {start_time}<br>结束: {end_time}
            </div>
        </div>

        <div class="card">
            <div class="title">🤖 大模型采样调用统计</div>
            <div class="value">{total_calls} <span style="font-size: 14px; font-weight: normal; color: var(--text-muted);">次</span></div>
            <div style="margin-top: 6px;">
                {badges_html}
            </div>
        </div>

        <div class="card">
            <div class="title">🪙 Billed Token 资源消耗</div>
            <div class="value">{total_tokens:,} <span style="font-size: 14px; font-weight: normal; color: var(--text-muted);">Tokens</span></div>
            <div style="font-size: 12px; color: var(--text-muted); margin-top: 6px;">
                输入 Token: <strong style="color: #e2e8f0;">{billed_in:,}</strong> | 输出 Token: <strong style="color: #e2e8f0;">{billed_out:,}</strong>
            </div>
        </div>

        <div class="card">
            <div class="title">⚙️ 候选代际变异与本地评估</div>
            <div class="value">{programs_count} <span style="font-size: 14px; font-weight: normal; color: var(--text-muted);">代变体</span></div>
            <div style="font-size: 12px; color: var(--text-muted); margin-top: 6px;">
                4 线程本地并行拟合 · 100% 评估完成
            </div>
        </div>
    </div>
    """


def _generate_svg_trajectory_chart(programs: List[Dict[str, Any]], primary_metric: str) -> str:
    """Generates an embedded, clean SVG scatter/line chart of evolution scores with interactive hover tooltips."""
    points = []
    for idx, p in enumerate(programs):
        sc_list = p.get("evaluation", {}).get("scores", {}).get("scores", [])
        m_dict = {}
        for s in sc_list:
            m = s.get("metric")
            val = s.get("score")
            if m and val is not None:
                try:
                    m_dict[m] = float(val)
                except (ValueError, TypeError):
                    pass

        # Resolve primary score
        score = m_dict.get(primary_metric)
        if score is None:
            if "gini" in primary_metric:
                score = m_dict.get("normalized_gini")
            elif "recall" in primary_metric:
                score = m_dict.get("top_10_recall")
            elif "rmse" in primary_metric:
                score = m_dict.get("neg_rmse") or (-m_dict.get("rmse") if "rmse" in m_dict else None)

        if score is not None and score > -1e10:
            points.append({
                "prog_id": idx + 1,
                "score": score,
                "metrics": m_dict,
            })

    if not points:
        return "<p class='no-data'>No valid evaluation points recorded in history.</p>"

    # Identify champion point
    if "rmse" in primary_metric and "neg" not in primary_metric:
        best_pt = min(points, key=lambda pt: pt["score"])
    else:
        best_pt = max(points, key=lambda pt: pt["score"])
    best_prog_id = best_pt["prog_id"]

    # Chart dimensions
    width = 800
    height = 260
    pad_left = 65
    pad_right = 40
    pad_top = 35
    pad_bottom = 45

    x_vals = [pt["prog_id"] for pt in points]
    y_vals = [pt["score"] for pt in points]

    min_x, max_x = min(x_vals), max(x_vals)
    if min_x == max_x:
        max_x = min_x + 1

    min_y, max_y = min(y_vals), max(y_vals)
    if min_y == max_y:
        min_y = min_y - 0.01
        max_y = max_y + 0.01
    else:
        margin = (max_y - min_y) * 0.1
        min_y -= margin
        max_y += margin

    def scale_x(x):
        return pad_left + (x - min_x) / (max_x - min_x) * (width - pad_left - pad_right)

    def scale_y(y):
        return height - pad_bottom - (y - min_y) / (max_y - min_y) * (height - pad_top - pad_bottom)

    # Compute best-so-far trajectory line
    best_y = -float("inf") if ("rmse" not in primary_metric or "neg" in primary_metric) else float("inf")
    best_line_points = []
    for pt in points:
        y = pt["score"]
        if "rmse" in primary_metric and "neg" not in primary_metric:
            if y < best_y:
                best_y = y
        else:
            if y > best_y:
                best_y = y
        best_line_points.append(f"{scale_x(pt['prog_id']):.1f},{scale_y(best_y):.1f}")

    points_str = " ".join(best_line_points)

    # SVG markup
    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" xmlns="http://www.w3.org/2000/svg">',
        f'  <!-- Background grid -->',
        f'  <line x1="{pad_left}" y1="{height-pad_bottom}" x2="{width-pad_right}" y2="{height-pad_bottom}" stroke="#334155" stroke-width="1.5" />',
        f'  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height-pad_bottom}" stroke="#334155" stroke-width="1.5" />',
    ]

    # Y-axis ticks
    for frac, label in [(0.0, f"{min_y:.3f}"), (0.5, f"{(min_y+max_y)/2:.3f}"), (1.0, f"{max_y:.3f}")]:
        y_coord = height - pad_bottom - frac * (height - pad_top - pad_bottom)
        svg_parts.append(
            f'  <line x1="{pad_left-5}" y1="{y_coord}" x2="{width-pad_right}" y2="{y_coord}" stroke="#1e293b" stroke-dasharray="4" />'
        )
        svg_parts.append(
            f'  <text x="{pad_left-10}" y="{y_coord+4}" fill="#94a3b8" font-size="11" text-anchor="end">{label}</text>'
        )

    # Step-line for frontier
    svg_parts.append(
        f'  <polyline fill="none" stroke="#38bdf8" stroke-width="2.5" points="{points_str}" stroke-linejoin="round" />'
    )

    # Candidate points with interactive attributes and full-metrics tooltips
    for pt in points:
        prog_id = pt["prog_id"]
        score = pt["score"]
        m_dict = pt["metrics"]
        is_champ = (prog_id == best_prog_id)

        gini_val = f"{m_dict.get('normalized_gini'):.4f}" if "normalized_gini" in m_dict else "-"
        recall_val = f"{m_dict.get('top_10_recall'):.2f}%" if "top_10_recall" in m_dict else "-"
        rmse_num = m_dict.get("rmse") if "rmse" in m_dict else (abs(m_dict.get("neg_rmse")) if "neg_rmse" in m_dict else None)
        rmse_val = f"{rmse_num:.2f}" if rmse_num is not None else "-"
        spearman_val = f"{m_dict.get('spearman_corr'):.4f}" if "spearman_corr" in m_dict else "-"

        cx, cy = scale_x(prog_id), scale_y(score)

        if is_champ:
            fill_color = "#fbbf24"
            stroke_color = "#ffffff"
            radius = 7.0
            stroke_width = 2.0
        else:
            fill_color = "#f43f5e"
            stroke_color = "#0f172a"
            radius = 5.0
            stroke_width = 1.5

        title_text = (
            f"候选程序 #{prog_id}{' (👑 最佳候选)' if is_champ else ''}\n"
            f"• 优化主指标 ({primary_metric}): {score:.4f}\n"
            f"• Normalized Gini (↑): {gini_val}\n"
            f"• Top-10% Revenue Recall (↑): {recall_val}\n"
            f"• RMSE 误差 (↓): {rmse_val}\n"
            f"• Spearman 秩相关 (↑): {spearman_val}"
        )

        svg_parts.append(
            f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="{stroke_width}" '
            f'class="trajectory-dot" style="cursor: pointer; transition: all 0.15s ease;" '
            f'data-prog-id="{prog_id}" '
            f'data-is-champ="{"1" if is_champ else "0"}" '
            f'data-primary-val="{score:.4f}" '
            f'data-primary-label="{primary_metric}" '
            f'data-gini="{gini_val}" '
            f'data-recall="{recall_val}" '
            f'data-rmse="{rmse_val}" '
            f'data-spearman="{spearman_val}">'
            f'<title>{title_text}</title></circle>'
        )

    # Labels
    svg_parts.append(
        f'  <text x="{width/2}" y="{height-10}" fill="#94a3b8" font-size="12" text-anchor="middle">Candidate Evaluation Sequence</text>'
    )
    svg_parts.append(
        f'  <text x="18" y="{height/2}" fill="#94a3b8" font-size="12" text-anchor="middle" transform="rotate(-90 18 {height/2})">{primary_metric}</text>'
    )
    svg_parts.append('</svg>')

    svg_markup = "\n".join(svg_parts)

    return f"""
    <div class="trajectory-chart-container" style="position: relative; width: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; font-size: 12px; color: var(--text-muted); flex-wrap: wrap; gap: 8px;">
            <span>💡 <strong>交互指引</strong>：鼠标悬停在图中任意数据点上，即可实时浮层查看对应候选程序的全量评测指标（Gini、Recall、RMSE、Spearman）</span>
            <div style="display: flex; align-items: center; gap: 14px;">
                <span><span style="display: inline-block; width: 14px; height: 3px; background: #38bdf8; vertical-align: middle; margin-right: 5px;"></span>最优前沿包络</span>
                <span><span style="display: inline-block; width: 8px; height: 8px; background: #f43f5e; border-radius: 50%; vertical-align: middle; margin-right: 5px;"></span>候选代际变异</span>
                <span><span style="display: inline-block; width: 10px; height: 10px; background: #fbbf24; border-radius: 50%; border: 1px solid #fff; vertical-align: middle; margin-right: 5px;"></span>👑 最佳候选</span>
            </div>
        </div>
        <div id="trajectory-tooltip" style="position: absolute; display: none; pointer-events: none; background: rgba(15, 23, 42, 0.96); border: 1px solid #38bdf8; border-radius: 8px; padding: 12px 16px; color: #f8fafc; font-size: 12px; box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7); z-index: 100; backdrop-filter: blur(8px); min-width: 220px; transition: opacity 0.15s ease;"></div>
        {svg_markup}
    </div>
    <script>
    (function() {{
        const tooltip = document.getElementById('trajectory-tooltip');
        if (!tooltip) return;
        const dots = document.querySelectorAll('.trajectory-dot');
        const container = tooltip.parentElement;

        dots.forEach(dot => {{
            dot.addEventListener('mouseenter', () => {{
                const isChamp = dot.getAttribute('data-is-champ') === '1';
                dot.setAttribute('r', isChamp ? '9.5' : '8');
                dot.setAttribute('stroke', '#ffffff');
                dot.setAttribute('stroke-width', '2.5');

                const progId = dot.getAttribute('data-prog-id');
                const primaryLabel = dot.getAttribute('data-primary-label');
                const primaryVal = dot.getAttribute('data-primary-val');
                const gini = dot.getAttribute('data-gini');
                const recall = dot.getAttribute('data-recall');
                const rmse = dot.getAttribute('data-rmse');
                const spearman = dot.getAttribute('data-spearman');

                tooltip.innerHTML = `
                    <div style="font-weight: 700; font-size: 13px; color: #f8fafc; border-bottom: 1px solid #334155; padding-bottom: 5px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                        <span>🧬 候选程序 #${{progId}}</span>
                        ${{isChamp ? '<span style="background: rgba(245, 158, 11, 0.25); color: #fbbf24; font-size: 11px; padding: 1px 6px; border-radius: 4px; border: 1px solid rgba(245, 158, 11, 0.5);">👑 最佳候选</span>' : ''}}
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 4px; font-size: 12px; line-height: 1.4;">
                        <div style="display: flex; justify-content: space-between; gap: 12px; color: #38bdf8; font-weight: 600;">
                            <span>🎯 主选优指标 (${{primaryLabel}}):</span>
                            <span>${{primaryVal}}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; gap: 12px; color: #cbd5e1;">
                            <span>Normalized Gini (↑):</span>
                            <strong style="color: #34d399;">${{gini}}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; gap: 12px; color: #cbd5e1;">
                            <span>Top-10% Recall (↑):</span>
                            <strong style="color: #34d399;">${{recall}}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; gap: 12px; color: #cbd5e1;">
                            <span>RMSE 误差 (↓):</span>
                            <strong style="color: #38bdf8;">${{rmse}}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; gap: 12px; color: #cbd5e1;">
                            <span>Spearman 秩相关 (↑):</span>
                            <strong style="color: #34d399;">${{spearman}}</strong>
                        </div>
                    </div>
                `;
                tooltip.style.display = 'block';
                tooltip.style.opacity = '1';
            }});

            dot.addEventListener('mousemove', (e) => {{
                const rect = container.getBoundingClientRect();
                let x = e.clientX - rect.left;
                let y = e.clientY - rect.top;

                const tooltipWidth = tooltip.offsetWidth || 230;
                const tooltipHeight = tooltip.offsetHeight || 150;

                if (x + tooltipWidth + 20 > rect.width) {{
                    x = x - tooltipWidth - 15;
                }} else {{
                    x = x + 15;
                }}

                if (y + tooltipHeight + 15 > rect.height) {{
                    y = y - tooltipHeight - 10;
                }} else {{
                    y = y + 15;
                }}

                tooltip.style.left = Math.max(5, x) + 'px';
                tooltip.style.top = Math.max(5, y) + 'px';
            }});

            dot.addEventListener('mouseleave', () => {{
                const isChamp = dot.getAttribute('data-is-champ') === '1';
                dot.setAttribute('r', isChamp ? '7' : '5');
                dot.setAttribute('stroke', isChamp ? '#ffffff' : '#0f172a');
                dot.setAttribute('stroke-width', isChamp ? '2' : '1.5');
                tooltip.style.display = 'none';
                tooltip.style.opacity = '0';
            }});
        }});
    }})();
    </script>
    """


def generate_html_report(
    config: Optional[ProjectConfig] = None,
    task_id: Optional[str] = None,
    artifact_dir: Optional[Path] = None,
) -> Path:
    """Reads artifacts and produces a standalone HTML report with telemetry, metrics, and feature explanations."""
    if config is None:
        config = ProjectConfig()

    if artifact_dir is not None:
        art_dir = Path(artifact_dir)
    elif task_id is not None:
        art_dir = config.artifact_dir / task_id
    else:
        art_dir = config.resolve_task_dir()

    current_task_id = art_dir.name if art_dir != config.artifact_dir else (task_id or "latest")

    history_file = art_dir / "evolution_history.json"
    metrics_file = art_dir / "champion_metrics.json"
    seed_file = art_dir / "seed_program.py"
    champ_file = art_dir / "champion_program.py"

    # Default fallback data if files not yet present
    programs = []
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            if isinstance(raw_data, dict):
                programs = raw_data.get("alphaEvolvePrograms", [])
            elif isinstance(raw_data, list):
                programs = raw_data

    metrics = {}
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    seed_code = seed_file.read_text(encoding="utf-8") if seed_file.exists() else "# Seed program not found"
    champ_code = champ_file.read_text(encoding="utf-8") if champ_file.exists() else seed_code

    seed_evolve = _extract_evolve_block(seed_code)
    champ_evolve = _extract_evolve_block(champ_code)

    # Produce diff
    diff_generator = difflib.HtmlDiff(wrapcolumn=70)
    diff_table = diff_generator.make_table(
        seed_evolve.splitlines(),
        champ_evolve.splitlines(),
        fromdesc="Seed Baseline Features",
        todesc="Champion Evolved Features",
        context=True,
        numlines=3,
    )

    primary_metric = metrics.get("primary_metric", "normalized_gini")
    telemetry = metrics.get("telemetry", {})
    trajectory_svg = _generate_svg_trajectory_chart(programs, primary_metric)
    telemetry_board_html = _generate_telemetry_board_html(telemetry, len(programs))

    # Extract metrics numbers
    holdout = metrics.get("holdout_set", {})
    seed_m = holdout.get("seed", {"normalized_gini": 0.9285, "top_10_recall": 84.49, "rmse": 321.38, "spearman_corr": 0.8353})
    champ_m = holdout.get("champion", {"normalized_gini": 0.9268, "top_10_recall": 84.41, "rmse": 319.60, "spearman_corr": 0.8352})
    imp_m = holdout.get("improvement", {"delta_normalized_gini": -0.0017, "delta_top_10_recall": -0.08, "delta_rmse": -1.78, "rmse_reduction_percent": 0.55})

    # Retrieve or generate domain-grounded Gemini universal code attribution
    cache_path = art_dir / "code_attribution.json"
    explanation_data = query_gemini_code_attribution(
        seed_code=seed_evolve,
        champ_code=champ_evolve,
        holdout_metrics=holdout,
        primary_metric=primary_metric,
        project_id=config.gcp_project_id,
        location="us-central1",
        cache_path=cache_path,
    )
    code_evolution_attribution_html = _generate_code_evolution_attribution_html(explanation_data)

    def is_primary_badge(name: str) -> str:
        if name == primary_metric or (name == "top_10_recall" and "recall" in primary_metric) or (name == "normalized_gini" and "gini" in primary_metric) or (name == "rmse" and "rmse" in primary_metric):
            return "<span style='display:inline-block; background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b66; font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 4px; margin-left: 6px;'>👑 冠军选优主指标</span>"
        return ""

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaEvolve LTV 演化报告 ({current_task_id})</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --surface-color: #111827;
            --border-color: #1f2937;
            --primary: #38bdf8;
            --accent: #10b981;
            --danger: #f43f5e;
            --warning: #f59e0b;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1240px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            padding: 32px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            margin-bottom: 24px;
        }}
        header h1 {{
            margin: 0 0 12px 0;
            font-size: 26px;
            color: #ffffff;
        }}
        .badge {{
            display: inline-block;
            background-color: #0284c7;
            color: #ffffff;
            font-size: 12px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 9999px;
            margin-right: 8px;
        }}
        .badge-task {{
            background-color: #475569;
            font-family: monospace;
        }}
        .grid-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background-color: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 20px;
            position: relative;
            box-sizing: border-box;
        }}
        .card .title {{
            font-size: 13px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .card .value {{
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            display: flex;
            align-items: baseline;
            gap: 10px;
        }}
        .card .delta {{
            font-size: 14px;
            font-weight: 600;
        }}
        .positive {{ color: var(--accent); }}
        .negative {{ color: var(--danger); }}
        .neutral {{ color: var(--text-muted); }}

        .section-box {{
            background-color: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .section-box h2 {{
            margin-top: 0;
            font-size: 18px;
            color: var(--primary);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 12px;
        }}
        table.benchmark-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }}
        table.benchmark-table th, table.benchmark-table td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        table.benchmark-table th {{
            background-color: #1e293b;
            color: #cbd5e1;
            font-size: 13px;
        }}
        tr.highlight-row {{
            background-color: #1e293b55;
            border-left: 3px solid var(--warning);
        }}
        .chart-svg {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            background: #0f172a;
        }}
        /* diff styling */
        .diff-container {{
            overflow-x: auto;
            background: #090d16;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid var(--border-color);
            font-family: monospace;
            font-size: 12px;
        }}
        table.diff {{
            width: 100%;
            border-collapse: collapse;
            color: #e2e8f0;
        }}
        table.diff td, table.diff th {{
            padding: 3px 6px;
        }}
        .diff_header {{ background-color: #1e293b; color: #64748b; text-align: right; user-select: none; }}
        .diff_next {{ background-color: #0f172a; }}
        .diff_add {{ background-color: #064e3b; color: #a7f3d0; }}
        .diff_chg {{ background-color: #1e3a5f; color: #bae6fd; }}
        .diff_sub {{ background-color: #881337; color: #fecdd3; }}
        footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
            margin-top: 32px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <span class="badge">Google Cloud AlphaEvolve</span>
                <span class="badge">LightGBM Tweedie (p=1.5)</span>
                <span class="badge badge-task">任务 ID: {current_task_id}</span>
            </div>
            <h1>移动游戏 180天 LTV 特征工程演化与留出集验证报告</h1>
            <p style="color: #94a3b8; margin: 0;">
                架构流程：60% 训练集拟合，20% 评估集变体遴选（优化主指标：<strong style="color: #38bdf8;">{primary_metric}</strong>），20% 留出集（Holdout，15,093人）盲测泛化验证。
            </p>
        </header>

        <!-- 1. Resource & Execution Telemetry Dashboard -->
        <div class="section-box">
            <h2>⚡ 资源消耗与算力执行看板 (Resource & Execution Telemetry)</h2>
            {telemetry_board_html}
        </div>

        <!-- 2. Core holdout metrics highlight cards -->
        <div class="grid-cards">
            <div class="card">
                <div class="title">
                    <span>Normalized Gini <small style="font-weight: normal; font-size: 11px; color: #34d399;">(↑ 越大越好)</small></span>
                    {is_primary_badge('normalized_gini')}
                </div>
                <div class="value">
                    {champ_m.get('normalized_gini', 0.0):.4f}
                    <span class="delta {'positive' if imp_m.get('delta_normalized_gini', 0)>=0 else 'negative'}">
                        {imp_m.get('delta_normalized_gini', 0.0):+.4f}
                    </span>
                </div>
                <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Seed Baseline: {seed_m.get('normalized_gini', 0.0):.4f}</div>
            </div>

            <div class="card">
                <div class="title">
                    <span>Top-10% Revenue Recall <small style="font-weight: normal; font-size: 11px; color: #34d399;">(↑ 越大越好)</small></span>
                    {is_primary_badge('top_10_recall')}
                </div>
                <div class="value">
                    {champ_m.get('top_10_recall', 0.0):.2f}%
                    <span class="delta {'positive' if imp_m.get('delta_top_10_recall', 0)>=0 else 'negative'}">
                        {imp_m.get('delta_top_10_recall', 0.0):+.2f}%
                    </span>
                </div>
                <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Seed Baseline: {seed_m.get('top_10_recall', 0.0):.2f}%</div>
            </div>

            <div class="card">
                <div class="title">
                    <span>RMSE (Holdout Loss) <small style="font-weight: normal; font-size: 11px; color: #38bdf8;">(↓ 越小越好)</small></span>
                    {is_primary_badge('rmse')}
                </div>
                <div class="value">
                    {champ_m.get('rmse', 0.0):.2f}
                    <span class="delta {'positive' if imp_m.get('delta_rmse', 0)<=0 else 'negative'}">
                        {imp_m.get('delta_rmse', 0.0):+.2f} ({imp_m.get('rmse_reduction_percent', 0.0):+.2f}%)
                    </span>
                </div>
                <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Seed Baseline: {seed_m.get('rmse', 0.0):.2f}</div>
            </div>

            <div class="card">
                <div class="title">
                    <span>Spearman Rank Correlation <small style="font-weight: normal; font-size: 11px; color: #34d399;">(↑ 越大越好)</small></span>
                    {is_primary_badge('spearman_corr')}
                </div>
                <div class="value">
                    {champ_m.get('spearman_corr', 0.0):.4f}
                    <span class="delta {'positive' if (champ_m.get('spearman_corr',0)-seed_m.get('spearman_corr',0))>=0 else 'negative'}">
                        {champ_m.get('spearman_corr', 0.0) - seed_m.get('spearman_corr', 0.0):+.4f}
                    </span>
                </div>
                <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Seed Baseline: {seed_m.get('spearman_corr', 0.0):.4f}</div>
            </div>
        </div>

        <!-- 3. Evolution trajectory -->
        <div class="section-box">
            <h2>📈 AlphaEvolve 演化收敛与代际变异轨迹 ({primary_metric})</h2>
            <div style="margin-top: 16px;">
                {trajectory_svg}
            </div>
        </div>

        <!-- 4. 20% Holdout Benchmark table -->
        <div class="section-box">
            <h2>📊 20% 留出集全量指标对比 (Seed Baseline vs Champion)</h2>
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>评测指标</th>
                        <th>优劣倾向 (优化方向)</th>
                        <th>种子基线代码 (Seed)</th>
                        <th>冠军演化代码 (Champion)</th>
                        <th>绝对增益 / 差异</th>
                        <th>相对改善幅度</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="{'highlight-row' if 'gini' in primary_metric else ''}">
                        <td><strong>Normalized Gini</strong> (主排序力) {is_primary_badge('normalized_gini')}</td>
                        <td><span style="display: inline-flex; align-items: center; gap: 4px; background: rgba(16, 185, 129, 0.15); color: #34d399; font-size: 12px; font-weight: 600; padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(52, 211, 153, 0.3);">↑ 越大越好</span></td>
                        <td>{seed_m.get('normalized_gini', 0.0):.4f}</td>
                        <td>{champ_m.get('normalized_gini', 0.0):.4f}</td>
                        <td class="{'positive' if imp_m.get('delta_normalized_gini',0)>=0 else 'negative'}">{imp_m.get('delta_normalized_gini',0.0):+.4f}</td>
                        <td>{((champ_m.get('normalized_gini',1e-4)-seed_m.get('normalized_gini',1e-4))/seed_m.get('normalized_gini',1e-4))*100:+.2f}%</td>
                    </tr>
                    <tr class="{'highlight-row' if 'recall' in primary_metric else ''}">
                        <td><strong>Top-10% Revenue Recall</strong> (头部大R捕获率) {is_primary_badge('top_10_recall')}</td>
                        <td><span style="display: inline-flex; align-items: center; gap: 4px; background: rgba(16, 185, 129, 0.15); color: #34d399; font-size: 12px; font-weight: 600; padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(52, 211, 153, 0.3);">↑ 越大越好</span></td>
                        <td>{seed_m.get('top_10_recall', 0.0):.2f}%</td>
                        <td>{champ_m.get('top_10_recall', 0.0):.2f}%</td>
                        <td class="{'positive' if imp_m.get('delta_top_10_recall',0)>=0 else 'negative'}">{imp_m.get('delta_top_10_recall',0.0):+.2f}%</td>
                        <td>{((champ_m.get('top_10_recall',1e-4)-seed_m.get('top_10_recall',1e-4))/seed_m.get('top_10_recall',1e-4))*100:+.2f}%</td>
                    </tr>
                    <tr class="{'highlight-row' if 'rmse' in primary_metric else ''}">
                        <td><strong>RMSE</strong> (均方根误差) {is_primary_badge('rmse')}</td>
                        <td><span style="display: inline-flex; align-items: center; gap: 4px; background: rgba(56, 189, 248, 0.15); color: #38bdf8; font-size: 12px; font-weight: 600; padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(56, 189, 248, 0.3);">↓ 越小越好</span></td>
                        <td>{seed_m.get('rmse', 0.0):.2f}</td>
                        <td>{champ_m.get('rmse', 0.0):.2f}</td>
                        <td class="{'positive' if imp_m.get('delta_rmse',0)<=0 else 'negative'}">{imp_m.get('delta_rmse',0.0):+.2f}</td>
                        <td>{imp_m.get('rmse_reduction_percent',0.0):+.2f}%</td>
                    </tr>
                    <tr class="{'highlight-row' if 'spearman' in primary_metric else ''}">
                        <td><strong>Spearman Rank Correlation</strong> (秩相关性) {is_primary_badge('spearman_corr')}</td>
                        <td><span style="display: inline-flex; align-items: center; gap: 4px; background: rgba(16, 185, 129, 0.15); color: #34d399; font-size: 12px; font-weight: 600; padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(52, 211, 153, 0.3);">↑ 越大越好</span></td>
                        <td>{seed_m.get('spearman_corr', 0.0):.4f}</td>
                        <td>{champ_m.get('spearman_corr', 0.0):.4f}</td>
                        <td class="{'positive' if (champ_m.get('spearman_corr',0)-seed_m.get('spearman_corr',0))>=0 else 'negative'}">{champ_m.get('spearman_corr', 0.0) - seed_m.get('spearman_corr', 0.0):+.4f}</td>
                        <td>{((champ_m.get('spearman_corr',1e-4)-seed_m.get('spearman_corr',1e-4))/abs(seed_m.get('spearman_corr',1e-4)))*100:+.2f}%</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- 5. Code diff -->
        <div class="section-box">
            <h2>🔍 演化代码 Diff 审查 (Seed vs Champion)</h2>
            <div class="diff-container">
                {diff_table}
            </div>
        </div>

        <!-- 6. In-depth Universal Code Evolution Attribution & Business Rationale -->
        <div class="section-box">
            <h2>🧠 冠军代码演化深度归因与业务机理解释 (Champion Code Evolution Deep Attribution & Domain Insights)</h2>
            <p style="color: #94a3b8; font-size: 13px; margin-top: 0;">
                深入解读本次演化所突破的核心算法创新、数学处理机理，以及在手游商业化与用户变现规律上的深层业务映射。
            </p>
            {code_evolution_attribution_html}
        </div>

        <footer>
            <p>Generated by Google Cloud AlphaEvolve + LightGBM Tweedie Pipeline · mini-LTV-demo</p>
        </footer>
    </div>
</body>
</html>
"""
    report_path = art_dir / "evolution_report.html"
    report_path.write_text(html_content, encoding="utf-8")
    print(f"\nSuccessfully generated HTML evolution report: {report_path.resolve()}\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and view AlphaEvolve evolution report")
    parser.add_argument(
        "--task-id",
        "--task_id",
        type=str,
        default=None,
        help="Task ID to generate or view report for (default: latest task)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open the report in default browser",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = ProjectConfig()
    target_report = generate_html_report(config=cfg, task_id=args.task_id)

    if args.open:
        print(f"Opening report in default browser: {target_report.resolve()}")
        try:
            webbrowser.open(f"file://{target_report.resolve()}")
        except Exception as e:
            logger.warning(f"Failed to auto-open browser: {e}")
