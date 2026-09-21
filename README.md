# Mobile Game LTV Forecasting Challenge - AlphaEvolve 特征工程演化优化系统

本项目基于 Kaggle 竞赛 [Mobile Game LTV Forecasting Challenge](https://www.kaggle.com/competitions/mobile-game-ltv-forecasting-challenge) 的全量用户数据，使用 Google Cloud **AlphaEvolve**（基于 Gemini 大模型驱动的演化编码引擎）对移动游戏 180 天 LTV 预测的**特征工程**进行自动化搜索与演化优化。

模型底层采用冻结的 **LightGBM (Tweedie分布, $`p=1.5`$)** 工业级基线。系统仅在特征工程代码块（标注 `# EVOLVE-BLOCK`）内进行搜索演化，在保持模型结构与超参数不变的前提下，持续挖掘业务变现动量、用户留存效率与长尾大额付费（Whale）的特征交互。

> 📘 **架构设计与理论探索**：
> - 完整系统设计方案与模块图谱详见 [DESIGN.md](DESIGN.md)。
> - 历史模型选型实验与 Tweedie 分布数学归因详见 [INSIGHT.md](INSIGHT.md)。

---

## 1. 目录结构

```text
.
├── Makefile                     # 自动化命令入口 (setup, auth, split, run, report, test)
├── config.yaml                  # 全局唯一定义文件 (GCP 凭据、演化超参、模型配比、数据配置)
├── requirements.txt             # 项目依赖包清单
├── DESIGN.md                    # 详细架构与技术设计文档
├── INSIGHT.md                   # 历史多模型探索、Tweedie 数学机理与决策归因报告
├── pytest.ini                   # Pytest 运行配置
├── venv/                        # Python 3.11 独立虚拟环境
├── alpha_evolve/                # Google Cloud 官方 AlphaEvolve 客户端与控制器 (严格只读)
├── src/                         # 核心业务代码库 (仿照官方 examples 规范)
│   ├── program.py               # 种子代码与可变特征工程块 (# EVOLVE-BLOCK)
│   ├── domain_context.py        # 手游 LTV 商业化领域知识库与 Gemini 动态特征解释引擎
│   ├── evaluate.py              # 本地候选代码评估 Harness (沙箱执行、并行线程池拟合、容错诊断)
│   ├── run_evolution.py         # 演化总控入口 (生命周期管理、控制器循环、留出集对比)
│   ├── report.py                # 交互式 HTML 演化成果报告生成器
│   ├── config.py                # Python 端轻量配置映射与动态库导入处理
│   ├── dataset.py               # 数据加载、三向科学分层切分 (60%/20%/20%)、原始特征函数
│   ├── models.py                # LightGBM Tweedie (p=1.5) 冻结模型类
│   └── metrics.py               # 工业级评测指标库 (Normalized Gini, Top-10% Recall, RMSE 等)
├── data/                        # 数据目录
│   ├── train_wide.csv           # [全量宽表] 75,464 用户 × 64 列原始指标
│   ├── train_split.csv          # [60% 训练集] 45,278 样本，用于各轮变体拟合
│   ├── eval_split.csv           # [20% 评估集] 15,093 样本，用于变体打分与冠军筛选
│   └── holdout_split.csv        # [20% 留出集] 15,093 样本，严格隔离用于最终 Seed vs Champion 盲测
├── artifacts/                   # 演化产物与报告目录 (按 task_id 独立隔离)
│   ├── <task_id>/               # 单次任务独立归档目录 (如 task_20260909_172048)
│   │   ├── seed_program.py      # 初始种子特征代码备份
│   │   ├── champion_program.py  # 演化出的最优冠军特征代码
│   │   ├── evolution_history.json # 全代际候选程序变体与评估打分历史
│   │   ├── champion_metrics.json  # 20% 留出集最终对比指标与 GCP 资源消耗遥测数据
│   │   ├── feature_explanation.json # 本地持久化归因缓存 (供 make report 0ms 极速加载)
│   │   └── evolution_report.html  # 独立可视化的 HTML 演化全景报告
│   ├── latest -> <task_id>      # 指向最近一次任务的符号链接
│   └── latest_task.txt          # 记录最近一次任务 ID 的文本指针
└── tests/                       # 自动化测试脚本目录
    ├── test_config.py           # 配置文件与指标归一化、路径校验测试
    ├── test_dataset.py          # 三向切分分布一致性与特征测试
    ├── test_evaluator.py        # 种子执行与候选评估器容错与全指标测试
    └── test_report.py           # 领域知识库、Gemini 归因与报告生成测试
```

---

## 2. 前置准备与环境设置

### 2.1 基础软件要求
- **Python**: Python 3.11+
- **GNU Make**: 自动化工作流工具 (`make --version`)
- **Google Cloud SDK**: `gcloud` CLI（用于 GCP 凭据鉴权与 API 交互）

### 2.2 创建 Python 虚拟环境与安装依赖 (首次运行必须执行)
> [!IMPORTANT]
> **交付前置说明**：为避免依赖冲突与体积冗余，本项目交付时不包含 `venv/` 虚拟环境目录。在您首次运行任何演化任务或测试前，**必须先在项目根目录下创建虚拟环境并安装依赖**！
>
> 提供了以下两种方式（任选其一即可）：
>
> **方式一：通过 Makefile 一键自动化初始化（强烈推荐）**
> ```bash
> make setup
> ```
> 该命令会自动检测并在当前根目录下创建 `venv` 虚拟环境、升级 pip 并安装 `requirements.txt` 中的全部依赖。
>
> **方式二：手动命令行分步执行**
> ```bash
> # 1. 确保使用 Python 3.11 或以上版本创建虚拟环境
> python3.11 -m venv venv
>
> # 2. 激活虚拟环境
> source venv/bin/activate
>
> # 3. 升级 pip 并安装项目核心依赖
> pip install --upgrade pip
> pip install -r requirements.txt
> ```

### 2.3 GCP 账号与 Gemini Enterprise APP 设置
1. **获取 Google Cloud 项目权限**：
   - 准备一个已开通结算账号的 Google Cloud Project（记下您的 `PROJECT_ID`）。
2. **启用 Discovery Engine API**：
   ```bash
   gcloud services enable discoveryengine.googleapis.com --project=<YOUR_GCP_PROJECT_ID>
   ```
3. **获取 Gemini Enterprise APP ID**：
   - 在 Google Cloud Console 的 Gemini Enterprise 控制台中创建或查看应用，获取 Engine / App ID（记下您的 `GE_APP_ID`）。
4. **IAM 权限配置**：
   - 运行账号需具备 **Discovery Engine Editor** (`roles/discoveryengine.editor`) 或管理员角色。
   - 官方权限设置指导可参考：[AlphaEvolve 环境与 API 访问设置文档](https://docs.cloud.google.com/gemini/enterprise/docs/alphaevolve/developer-guide/environment-and-api-access-setup?hl=zh-cn)。

---

## 3. 全局配置文件 (`config.yaml`)

项目所有设置统一在根目录 `config.yaml` 中管理，无隐藏配置文件。首次运行时请在 `config.yaml` 中填入您自己的 GCP 凭据：

```yaml
# 1. Google Cloud 凭据与服务配置
gcp:
  project_id: "<YOUR_GCP_PROJECT_ID>"   # 请在此填入您的 Google Cloud Project ID
  location: "global"
  collection: "default_collection"
  ge_app_id: "<YOUR_GE_APP_ID>"         # 请在此填入您的 Gemini Enterprise App/Engine ID
  assistant: "default_assistant"
  base_url: "discoveryengine.googleapis.com"
  alpha_evolve_path: "./alpha_evolve"   # 支持指定为外部路径

# 2. 演化超参数
evolution:
  title: "Mobile Game LTV Feature Engineering Evolution"
  problem_description: "Evolve feature engineering transformations for 180-day mobile game LTV forecasting."
  program_language: "python"
  max_programs_generated: 20       # 默认生成上限
  max_programs_evaluated: 20       # 默认评估上限
  concurrency: 4                   # 云端生成并发度
  worker_concurrency: 4            # 本地并行评估线程数
  parallel_evaluation: true        # 启用本地并行线程池评估
  idle_timeout_s: 120              # 超时保护 (秒)
  primary_metric: "normalized_gini" # 核心遴选指标 (支持: normalized_gini, top_10_recall, neg_rmse, spearman_corr)

# 3. 大模型混合配比
models:
  - name: "gemini-3.5-flash"
    weight: 0.70
  - name: "gemini-3.1-pro-preview"
    weight: 0.30

# 4. 数据集三向切分
dataset:
  raw_wide_path: "./data/train_wide.csv"
  train_split_path: "./data/train_split.csv"
  eval_split_path: "./data/eval_split.csv"
  holdout_split_path: "./data/holdout_split.csv"
  train_ratio: 0.60
  eval_ratio: 0.20
  holdout_ratio: 0.20
  random_state: 42
  split_method: "stratified"
```

---

## 4. Makefile 核心工作流与命令

| 命令 | 说明与执行行为 |
| :--- | :--- |
| **`make setup`** | 创建 Python 3.11 `venv` 虚拟环境并自动安装所有依赖包（首次必须执行）。 |
| **`make auth`** | 调起浏览器执行 `gcloud auth application-default login` 完成 ADC 鉴权。 |
| **`make split`** | 将 `data/train_wide.csv` 按照 60% / 20% / 20% 分层切分为训练集、评估集、留出集。 |
| **`make run`** | 启动演化任务，自动分配 `task_id`，隔离产物至 `artifacts/<task_id>/`，结束时**自动生成 HTML 报告**。 |
| **`make run programs=2`** | **测试覆盖运行**：动态将代际预算覆盖为 2，快速跑通闭环。 |
| **`make run metric=recall`** | **动态指定选优主指标**：支持 `gini`、`recall`、`rmse`、`spearman`，指导 Gemini 变异并在结束时按该指标遴选冠军。 |
| **`make run task_id=custom`** | 指定自定义任务编号保存产物。 |
| **`make report`** | **自动在默认浏览器中打开**最新一次任务的 HTML 全景报告（支持传参 `make report task_id=...` 打开指定历史报告）。 |
| **`make test`** | 运行 `tests/` 目录下的自动化单元与集成测试。 |
| **`make mask`** | **项目交付脱敏**：一键将全项目配置与文档中的个人 `project_id`、`ge_app_id` 脱敏遮盖为占位符，并安全备份至本地 `.credentials.backup`。 |
| **`make mask-dry`** | **脱敏预览**：预览脱敏匹配的文件与出现频次（Dry-run），不实际修改文件。 |
| **`make unmask`** | **凭据恢复**：一键从本地 `.credentials.backup` 自动恢复凭据，或指定自定义凭据恢复。 |
| **`make clean`** | 清理 Python 编译缓存文件与临时数据。 |

---

## 5. 演化机制与报告特性

### 5.1 代码演化区域 (`# EVOLVE-BLOCK`)
在 `src/program.py` 中，仅被 `# EVOLVE-BLOCK-START` 与 `# EVOLVE-BLOCK-END` 包裹的 `engineer_features(df)` 函数会被云端 Gemini 演化；外部的 LightGBM Tweedie 拟合逻辑与评估接口保持冻结：

```python
# EVOLVE-BLOCK-START
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 变体特征工程演化区域...
    return df
# EVOLVE-BLOCK-END
```

### 5.2 三向科学切分隔离机制
1. **训练集 (60%, 45,278 样本)**：各候选变体在训练集上拟合 LightGBM Tweedie 模型。
2. **评估集 (20%, 15,093 样本)**：在评估集上计算指标，指导 AlphaEvolve 的亲本采样并挑选 Champion 冠军代码。
3. **留出集 (20%, 15,093 样本)**：全生命周期完全物理隔离，仅在演化结束时对 Seed 与 Champion 代码进行无偏对比，验证泛化增益。

### 5.3 交互式 HTML 报告核心看板
任务执行产出的 `artifacts/<task_id>/evolution_report.html` 具备以下深度特性：
- **⚡ 资源与算力执行看板**：展示任务总执行耗时、大模型调用总次数及按模型拆分次数（`gemini-3.5-flash` vs `gemini-3.1-pro-preview`）、Google Cloud API 真实 `inputTokenCount` 与 `outputTokenCount` 资源统计。
- **👑 冠军选优主指标高亮与优劣倾向标定**：
  - 对选优主指标赋予专属光环标记与排名对比。
  - 在「20% 留出集全量指标对比」表格中新增专门的「**优劣倾向 (优化方向)**」列，明确标定指标是 `↑ 越大越好` 还是 `↓ 越小越好`，顶部指标卡片同步提示优化方向，业务含义一目了然。
- **📈 演化收敛与代际变异轨迹图（交互式 Hover 全量指标卡片）**：
  - 展示全部候选程序的生成序列与最优前沿包络阶梯线，直观呈现变异探索与收敛过程。
  - **交互体验**：鼠标悬停在轨迹图中任意数据点上，数据点平滑放大高亮（普通点放大至 8px，👑 冠军点放大至 9.5px 并带金色光环），并跟随鼠标实时弹出毛玻璃悬浮卡片。
  - **全量指标多维展示**：浮层不仅展示当前选优主指标，还同步呈现其他全部维度的评估指标：
    - `Normalized Gini (↑)`
    - `Top-10% Revenue Recall (↑)`
    - `RMSE 误差 (↓)`（自动还原为业务真实绝对金额）
    - `Spearman 秩相关 (↑)`
  - 内置自适应屏幕防溢出碰撞检测与原生 SVG `<title>` 脱机打印兼容。
- **🔍 演化代码 Diff 审查**：直观展示冠军代码与种子基线代码在 `# EVOLVE-BLOCK` 内的代码级 Diff 对比。
- **🧠 冠军代码演化深度归因与业务机理解释 (Universal Attribution Engine)**：
  - **超越单一特征工程**：自适应感知特征工程、模型超参数体系、自定义损失与目标函数、数据预处理采样等任意算法组件的演化。
  - **结合领域知识动态归因**：结合移动游戏 F2P 商业化领域知识库（180天 LTV 目标、D0~D7 观察窗、重尾零膨胀分布）由 Gemini 动态归因，深入剖析数学算法机理与商业化价值。
  - **架构精简与战略启示**：深度分析代码剪枝与基线保留情况，提炼 3 大工业级算法迭代与业务落地战略启示。
  - **本地秒级缓存与离线兜底**：归因结果自动持久化至 `code_attribution.json`，后续查看 0.05 秒秒级直出；支持无网/无凭据环境优雅降级。

## 6. 工业级评测指标体系与商业化机理解析

在移动游戏 180 天 LTV 预测场景中，四大评估指标互为补充，系统在报表与选优中均明确标定其优化方向与业务价值：

| 指标名称 | 优劣倾向 | 算法/统计定义 | 手游商业化与业务决策意义 |
| :--- | :---: | :--- | :--- |
| **Normalized Gini**<br>(归一化基尼系数) | **`↑ 越大越好`**<br>(值域 $`[0, 1]`$) | 衡量预测值累积分布相对于完美排序曲线的洛伦兹覆盖面积（即 $`\mathrm{Gini}(y, \hat{y}) / \mathrm{Gini}(y, y)`$）。 | **宏观用户分层与 UA 买量出价的核心排序力指标**。对单个极端大额偶发充值具有极佳的秩鲁棒性，不会因单个离群点而扭曲全量大盘评估。 |
| **Top-10% Revenue Recall**<br>(头部核心大R营收召回率) | **`↑ 越大越好`**<br>(百分比) | 预测 LTV 最高的头部 10% 用户，其实际充值金额占留出集全量实际总收入的百分比。 | **游戏变现极度依赖的“巨鲸”捕获率**。F2P 手游前 10% 核心付费者通常贡献 70%~85% 收入，直接决定精细化 VIP 运营与大额高意愿充值人群圈选的精准度。 |
| **RMSE**<br>(均方根误差) | **`↓ 越小越好`**<br>(美元绝对金额) | 实际真实累积收入与预测收入之间差值的平方和均值的平方根：$`\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}`$。 | **总财务流水预测误差测度**。由于 LTV 呈现严重的帕累托长尾，巨鲸误差平方占比较大。*注：在云端优化器中以 `neg_rmse`（负数，越大越好）形式打分，报告中自动还原为正向真实金额（越小越好）。* |
| **Spearman Rank Correlation**<br>(斯皮尔曼等级相关系数) | **`↑ 越大越好`**<br>(值域 $`[-1, 1]`$) | 预测值排名与真实值排名之间的皮尔逊相关系数：$`\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}`$。 | **纯粹单调排序能力**。完全不受非线性单调变换及极端异常值量纲影响，真实反映算法对玩家价值高低相对位次的辨别一致性。 |

---

## 7. 项目交付脱敏与隐私保护 (`make mask` & `make unmask`)

为保障企业级隐私安全，在将代码开源、分享给客户或推送到公共 GitHub 仓库前，请务必执行凭据脱敏工具。

### 7.1 脱敏命令
```bash
# 1. 预览模式 (查看哪些文件会被修改，不写盘)
make mask-dry

# 2. 正式执行脱敏 (自动替换 project_id 与 ge_app_id 为占位符，并保存本地恢复凭据)
make mask
```

脱敏工具会自动扫描代码库中的 `config.yaml`、文档和脚本，将您的真实凭证替换为 `<YOUR_GCP_PROJECT_ID>` 和 `<YOUR_GE_APP_ID>`，并在 `config.yaml` 中生成填写指导注释。同时会将您的原凭证安全暂存于本地 `.credentials.backup`（已被 `.gitignore` 保护，绝不会被推送到 Git）。

### 7.2 恢复凭证命令
```bash
# 1. 本地一键恢复 (自动读取 .credentials.backup 恢复为您本人的凭据):
make unmask

# 2. 或指定自定义凭证恢复:
make unmask PROJECT_ID=my-gcp-project-123 APP_ID=gemini-enterprise-999999
```

### 7.3 清理本地虚拟环境
```bash
rm -rf venv/
```
客户解压项目后，仅需依照本文档第 2 节运行 `make setup`（或手动 `python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`）即可秒级创建其本地环境并安装依赖。

