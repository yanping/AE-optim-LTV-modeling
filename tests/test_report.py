"""
Unit tests for the HTML report generator and AST feature explanation engine.
"""

from pathlib import Path
import pytest

from src.config import ProjectConfig
from src.report import (
    _analyze_feature_semantics,
    _extract_evolve_block,
    _extract_feature_assignments,
    _generate_feature_diff_analysis_html,
    _generate_svg_trajectory_chart,
    _generate_telemetry_board_html,
    generate_html_report,
)


def test_extract_feature_assignments_ast():
    code = """
# EVOLVE-BLOCK-START
df["f1"] = df["revenue_d0"] + 1.0
df["f2_multi_line"] = (
    df["revenue_d7"] - df["revenue_d0"]
)
# EVOLVE-BLOCK-END
"""
    assignments = _extract_feature_assignments(code)
    assert "f1" in assignments
    assert "f2_multi_line" in assignments
    assert "revenue_d0" in assignments["f1"]
    assert "revenue_d7" in assignments["f2_multi_line"]


from src.domain_context import (
    GAME_LTV_DOMAIN_KNOWLEDGE,
    query_gemini_code_attribution,
    query_gemini_feature_explanation,
    _build_fallback_attribution,
)
from src.report import (
    _generate_code_evolution_attribution_html,
    _generate_feature_diff_analysis_html,
)


def test_domain_context_knowledge():
    assert "Mobile Game 180-Day LTV" in GAME_LTV_DOMAIN_KNOWLEDGE
    assert "ltv_d8_d180" in GAME_LTV_DOMAIN_KNOWLEDGE
    assert "normalized_gini" in GAME_LTV_DOMAIN_KNOWLEDGE


def test_fallback_attribution_feature_engineering():
    seed = "# EVOLVE-BLOCK-START\ndf['base'] = df['d0']\n# EVOLVE-BLOCK-END"
    champ = "# EVOLVE-BLOCK-START\ndf['base'] = df['d0']\ndf['rev_trend'] = df['d7'] - df['d0']\n# EVOLVE-BLOCK-END"
    res = _build_fallback_attribution(seed, champ, "normalized_gini")
    assert res["source"] == "fallback"
    assert "特征工程" in res["evolution_scope"]
    assert len(res["key_innovations"]) == 1
    assert "rev_trend" in res["key_innovations"][0]["title"]
    assert len(res["strategic_insights"]) > 0


def test_fallback_attribution_model_config():
    seed = "params = {'learning_rate': 0.1}"
    champ = "params = {'learning_rate': 0.03, 'objective': 'tweedie'}"
    res = _build_fallback_attribution(seed, champ, "top_10_recall")
    assert res["source"] == "fallback"
    assert "模型架构" in res["evolution_scope"] or "算法" in res["evolution_scope"]
    assert len(res["key_innovations"]) >= 1


def test_generate_code_evolution_attribution_html():
    explanation_data = {
        "source": "gemini",
        "evolution_scope": "自定义目标函数与时序表示学习",
        "evolution_summary": "引入非对称 Tweedie 损失函数配合时序正交线性斜率，显著提升高净值玩家召回率。",
        "key_innovations": [
            {
                "title": "加权正交回归斜率",
                "type": "特征工程",
                "badge_color": "#38bdf8",
                "code_snippet": "df['rev_slope'] = (rev * weights).sum()",
                "technical_mechanism": "通过正交最小二乘系数提取全局一阶变化斜率。",
                "business_domain_impact": "识别观察期后程消费提速的大R鲸鱼。",
            },
            {
                "title": "非对称 Tweedie 损失",
                "type": "损失函数",
                "badge_color": "#10b981",
                "code_snippet": "def custom_tweedie(y_true, y_pred): ...",
                "technical_mechanism": "对头部大R预测偏差增加非对称梯度惩罚。",
                "business_domain_impact": "有效提升 Top-10% 巨鲸收入召回率。",
            }
        ],
        "simplifications_and_pruning": "原有基线逻辑无损保留，采取非破坏性正交扩展策略。",
        "strategic_insights": [
            "时序斜率特征显著提升了头部大R召回率",
            "决策树分裂未发生过拟合",
        ],
    }
    html = _generate_code_evolution_attribution_html(explanation_data)
    assert "加权正交回归斜率" in html
    assert "非对称 Tweedie 损失" in html
    assert "自定义目标函数与时序表示学习" in html
    assert "非破坏性正交扩展策略" in html
    assert "头部大R召回率" in html


def test_generate_telemetry_board_html():
    telemetry = {
        "duration_formatted": "12m 34s",
        "start_time": "2026-09-09 10:00:00",
        "end_time": "2026-09-09 10:12:34",
        "total_model_calls": 10,
        "model_calls_by_model": {
            "gemini-3.5-flash": {"calls": 7, "percentage": "70.0%"},
            "gemini-3.1-pro-preview": {"calls": 3, "percentage": "30.0%"},
        },
        "token_usage": {
            "billed_input_tokens": 12500,
            "billed_output_tokens": 34800,
            "total_tokens": 47300,
        },
    }
    board = _generate_telemetry_board_html(telemetry, 10)
    assert "12m 34s" in board
    assert "gemini-3.5-flash" in board
    assert "gemini-3.1-pro-preview" in board
    assert "47,300" in board
    assert "12,500" in board


def test_generate_html_report_cached_mode(tmp_path):
    import json
    task_dir = tmp_path / "task_20260909_cached"
    task_dir.mkdir(parents=True)

    seed_file = task_dir / "seed_program.py"
    seed_file.write_text("# EVOLVE-BLOCK-START\ndf['f1'] = 1\n# EVOLVE-BLOCK-END\n", encoding="utf-8")

    champ_file = task_dir / "champion_program.py"
    champ_file.write_text("# EVOLVE-BLOCK-START\ndf['f1'] = 1\ndf['rev_trend'] = 2\n# EVOLVE-BLOCK-END\n", encoding="utf-8")

    cache_file = task_dir / "code_attribution.json"
    cache_file.write_text(json.dumps({
        "source": "cached",
        "evolution_scope": "时序特征表示学习",
        "evolution_summary": "通过一阶差分捕捉早期跃迁",
        "key_innovations": [{
            "title": "rev_trend",
            "type": "特征工程",
            "badge_color": "#38bdf8",
            "code_snippet": "df['rev_trend'] = 2",
            "technical_mechanism": "差分计算",
            "business_domain_impact": "捕捉玩家早期付费跃迁"
        }],
        "simplifications_and_pruning": "基线逻辑全量保留",
        "strategic_insights": ["启示1：时序动量有效"]
    }), encoding="utf-8")

    cfg = ProjectConfig(artifact_dir=tmp_path, task_id="task_20260909_cached")
    report_path = generate_html_report(config=cfg, task_id="task_20260909_cached")
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "来自本地归因缓存" in content
    assert "rev_trend" in content
    assert "时序特征表示学习" in content


def test_generate_html_report_end_to_end(tmp_path):
    task_dir = tmp_path / "task_20260909_test"
    task_dir.mkdir(parents=True)

    seed_file = task_dir / "seed_program.py"
    seed_file.write_text("# EVOLVE-BLOCK-START\ndf['f1'] = 1\n# EVOLVE-BLOCK-END\n", encoding="utf-8")

    champ_file = task_dir / "champion_program.py"
    champ_file.write_text("# EVOLVE-BLOCK-START\ndf['f1'] = 1\ndf['rev_trend_projection'] = 2\n# EVOLVE-BLOCK-END\n", encoding="utf-8")

    cfg = ProjectConfig(artifact_dir=tmp_path, task_id="task_20260909_test")
    report_path = generate_html_report(config=cfg, task_id="task_20260909_test")

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "task_20260909_test" in content
    assert "rev_trend_projection" in content


def test_generate_svg_trajectory_chart_interactive_tooltips():
    programs = [
        {
            "createTime": "2026-09-10T05:50:00Z",
            "evaluation": {
                "scores": {
                    "scores": [
                        {"metric": "normalized_gini", "score": 0.9320},
                        {"metric": "top_10_recall", "score": 83.10},
                        {"metric": "neg_rmse", "score": -295.50},
                        {"metric": "spearman_corr", "score": 0.8315},
                    ]
                }
            },
        },
        {
            "createTime": "2026-09-10T05:51:00Z",
            "evaluation": {
                "scores": {
                    "scores": [
                        {"metric": "normalized_gini", "score": 0.9365},
                        {"metric": "top_10_recall", "score": 83.80},
                        {"metric": "neg_rmse", "score": -288.20},
                        {"metric": "spearman_corr", "score": 0.8340},
                    ]
                }
            },
        },
    ]
    html = _generate_svg_trajectory_chart(programs, "normalized_gini")
    assert "trajectory-chart-container" in html
    assert "trajectory-tooltip" in html
    assert "trajectory-dot" in html
    assert 'data-prog-id="1"' in html
    assert 'data-prog-id="2"' in html
    assert 'data-gini="0.9365"' in html
    assert 'data-recall="83.80%"' in html
    assert 'data-rmse="288.20"' in html
    assert 'data-spearman="0.8340"' in html
    assert 'data-is-champ="1"' in html
    assert "👑 最佳候选" in html
    assert "鼠标悬停在图中任意数据点上" in html
