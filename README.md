# Mobile Game LTV Forecasting Challenge - AlphaEvolve Feature Engineering Optimization System

**English** | [简体中文](README_CN.md)

This project benchmarks on the full user dataset from the Kaggle competition [Mobile Game LTV Forecasting Challenge](https://www.kaggle.com/competitions/mobile-game-ltv-forecasting-challenge), leveraging Google Cloud **AlphaEvolve** (Gemini-driven evolutionary coding engine) to autonomously discover, mutate, and optimize **feature engineering** pipelines for 180-day mobile game player lifetime value (LTV) forecasting.

The underlying model uses a frozen **LightGBM (Tweedie distribution, $`p=1.5`$)** industrial-grade baseline. The evolutionary search operates strictly within the designated feature engineering block (delimited by `# EVOLVE-BLOCK`), keeping model architecture and hyperparameters frozen while continuously discovering non-linear monetization momentum, player retention velocity, and long-tailed whale purchase interactions.

> 📘 **Architecture Design & Theoretical Exploration**:
> - For complete system architecture and module topology, see [DESIGN.md](DESIGN.md).
> - For baseline model benchmarking and Tweedie mathematical mechanism attribution, see [INSIGHT.md](INSIGHT.md).

---

## 1. Directory Structure

```text
.
├── Makefile                     # Automation workflow entry point (setup, auth, split, run, report, test, mask, unmask)
├── config.yaml                  # Global Single Source of Truth (GCP credentials, evolution parameters, models, data)
├── requirements.txt             # Python dependency manifest
├── README.md                    # User manual and technical guide (English)
├── README_CN.md                 # User manual and technical guide (简体中文)
├── DESIGN.md                    # Detailed architecture and system design specification (English)
├── DESIGN_CN.md                 # Detailed architecture and system design specification (简体中文)
├── INSIGHT.md                   # Modeling exploration, benchmarks, and Tweedie mathematical insights (English)
├── INSIGHT_CN.md                # Modeling exploration, benchmarks, and Tweedie mathematical insights (简体中文)
├── pytest.ini                   # Pytest test execution configuration
├── venv/                        # Python 3.11 isolated virtual environment
├── alpha_evolve/                # Google Cloud official AlphaEvolve client & controller (strictly read-only)
├── scripts/                     # Operational delivery & maintenance scripts
│   └── mask_credentials.py     # Automated credential sanitization and restoration tool
├── src/                         # Core implementation source code (modeled after official examples)
│   ├── program.py               # Seed program and mutable feature engineering block (# EVOLVE-BLOCK)
│   ├── domain_context.py        # F2P game commercial domain knowledge base & dynamic Gemini attribution engine
│   ├── evaluate.py              # Candidate program evaluator harness (sandbox execution, parallel thread pool, diagnostics)
│   ├── run_evolution.py         # Evolution master controller (lifecycle management, loop control, holdout validation)
│   ├── report.py                # Standalone interactive HTML report generator
│   ├── config.py                # Python lightweight configuration mapping and dynamic library loader
│   ├── dataset.py               # Data loading, balanced 3-way stratified partitioning (60%/20%/20%), base features
│   ├── models.py                # LightGBM Tweedie (p=1.5) frozen model wrapper
│   └── metrics.py               # Industrial evaluation metric library (Normalized Gini, Top-10% Recall, RMSE, Spearman)
├── data/                        # Dataset directory
│   ├── train_wide.csv           # [Full Wide Table] 75,464 players × 64 raw behavioural indicators
│   ├── train_split.csv          # [60% Train Split] 45,278 samples for model fitting across evolutionary generations
│   ├── eval_split.csv           # [20% Eval Split] 15,093 samples for candidate scoring and champion selection
│   └── holdout_split.csv        # [20% Holdout Split] 15,093 samples strictly isolated for unbiased Seed vs Champion blind testing
├── artifacts/                   # Output artifacts and evolution reports (isolated per task_id)
│   ├── <task_id>/               # Standalone task archive directory (e.g. task_20260909_172048)
│   │   ├── seed_program.py      # Seed baseline feature engineering backup
│   │   ├── champion_program.py  # Evolved optimal champion feature engineering code
│   │   ├── evolution_history.json # Full lineage of evaluated programs and generation scores
│   │   ├── champion_metrics.json  # Holdout comparison metrics and GCP token consumption telemetry
│   │   ├── feature_explanation.json # Local persistent attribution cache (powers instant make report rendering)
│   │   └── evolution_report.html  # Standalone interactive HTML evolution dashboard
│   ├── latest -> <task_id>      # Symlink to the most recent evolution run
│   └── latest_task.txt          # Text pointer recording the latest task ID
└── tests/                       # Automated test suite
    ├── test_config.py           # Configuration loading, metric normalization, and path validation tests
    ├── test_dataset.py          # 3-way split distribution consistency and feature integrity tests
    ├── test_evaluator.py        # Seed execution, candidate sandbox resilience, and all-metric tests
    ├── test_mask_credentials.py # Credential sanitization, backup, and restoration tests
    └── test_report.py           # Domain knowledge base, Gemini attribution, and HTML report tests
```

---

## 2. Prerequisites & Environment Setup

### 2.1 Software Prerequisites
- **Python**: Python 3.11+
- **GNU Make**: Workflow automation tool (`make --version`)
- **Google Cloud SDK**: `gcloud` CLI (for GCP credential authentication and API interactions)

### 2.2 Create Python Virtual Environment & Install Dependencies (Mandatory on First Run)
> [!IMPORTANT]
> **Delivery Prerequisite Notice**: To avoid dependency conflicts and packaging overhead, the `venv/` virtual environment directory is not included when the project is delivered. Before running any evolution tasks or test suites, **you must first initialize the virtual environment and install dependencies in the project root**!
>
> Two setup methods are supported (choose either):
>
> **Method 1: Automated One-Click Setup via Makefile (Recommended)**
> ```bash
> make setup
> ```
> This command automatically verifies Python 3.11+, creates the `venv` virtual environment in the repository root, upgrades pip, and installs all dependencies specified in `requirements.txt`.
>
> **Method 2: Manual Step-by-Step Setup**
> ```bash
> # 1. Create a virtual environment using Python 3.11 or above
> python3.11 -m venv venv
>
> # 2. Activate the virtual environment
> source venv/bin/activate
>
> # 3. Upgrade pip and install core project dependencies
> pip install --upgrade pip
> pip install -r requirements.txt
> ```

### 2.3 GCP Account & Gemini Enterprise App Configuration
1. **Google Cloud Project**:
   - Prepare a Google Cloud Project with active billing enabled (note down your `PROJECT_ID`).
2. **Enable Discovery Engine API**:
   ```bash
   gcloud services enable discoveryengine.googleapis.com --project=<YOUR_GCP_PROJECT_ID>
   ```
3. **Obtain Gemini Enterprise App ID**:
   - Create or locate your application in the Gemini Enterprise Console on Google Cloud (note down your `GE_APP_ID`).
4. **IAM Role Permissions**:
   - The authenticating account requires **Discovery Engine Editor** (`roles/discoveryengine.editor`) or Project Owner/Editor roles.
   - For official setup guidelines, see: [AlphaEvolve Environment and API Access Setup](https://docs.cloud.google.com/gemini/enterprise/docs/alphaevolve/developer-guide/environment-and-api-access-setup).

---

## 3. Global Configuration (`config.yaml`)

All system settings are consolidated in `config.yaml` located at the root of the repository. Before initial execution, insert your GCP credentials into `config.yaml`:

```yaml
# 1. Google Cloud Credentials & Service Settings
gcp:
  project_id: "<YOUR_GCP_PROJECT_ID>"   # Enter your Google Cloud Project ID here
  location: "global"
  collection: "default_collection"
  ge_app_id: "<YOUR_GE_APP_ID>"         # Enter your Gemini Enterprise App/Engine ID here
  assistant: "default_assistant"
  base_url: "discoveryengine.googleapis.com"
  alpha_evolve_path: "./alpha_evolve"   # Configurable relative or absolute library path

# 2. Evolution Hyperparameters
evolution:
  title: "Mobile Game LTV Feature Engineering Evolution"
  problem_description: "Evolve feature engineering transformations for 180-day mobile game LTV forecasting."
  program_language: "python"
  max_programs_generated: 20       # Default maximum program generation budget
  max_programs_evaluated: 20       # Default maximum program evaluation budget
  concurrency: 4                   # Cloud generation concurrency
  worker_concurrency: 4            # Local parallel evaluation worker threads
  parallel_evaluation: true        # Enable multi-threaded candidate evaluation
  idle_timeout_s: 120              # Timeout protection (seconds)
  primary_metric: "normalized_gini" # Primary metric for champion selection (supports: normalized_gini, top_10_recall, neg_rmse, spearman_corr)

# 3. Foundation Model Mixture Weights
models:
  - name: "gemini-3.5-flash"
    weight: 0.70
  - name: "gemini-3.1-pro-preview"
    weight: 0.30

# 4. Dataset 3-Way Balanced Partitioning
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

## 4. Makefile Workflows & Commands

| Command | Execution Behavior & Description |
| :--- | :--- |
| **`make setup`** | Create Python 3.11 `venv` and install all required dependencies (mandatory on first run). |
| **`make auth`** | Launch browser to execute `gcloud auth application-default login` for ADC credentials. |
| **`make split`** | Partition `data/train_wide.csv` into stratified 60% Train, 20% Eval, and 20% Holdout splits. |
| **`make run`** | Launch evolution task, auto-assign `task_id`, archive to `artifacts/<task_id>/`, and **automatically generate HTML report** upon completion. |
| **`make run programs=2`** | **Smoke Test Execution**: Dynamically override program generation budget to 2 to quickly verify end-to-end flow. |
| **`make run metric=recall`** | **Dynamic Primary Metric**: Supports `gini`, `recall`, `rmse`, `spearman` to guide Gemini mutations and select the champion. |
| **`make run task_id=custom`** | Assign a custom task identifier for artifact archiving. |
| **`make report`** | **Open in default browser**: Open the latest interactive HTML report (pass `make report task_id=...` for historical runs). |
| **`make test`** | Run automated unit and integration tests under `tests/`. |
| **`make mask`** | **Delivery Sanitization**: Anonymize personal `project_id` and `ge_app_id` with placeholders and back up original credentials to `.credentials.backup`. |
| **`make mask-dry`** | **Sanitization Preview**: Preview files and replacement counts without modifying disk. |
| **`make unmask`** | **Credentials Restore**: Restore credentials from local `.credentials.backup`, or supply custom credentials. |
| **`make clean`** | Clean Python bytecode caches and temporary files. |

---

## 5. Evolution Architecture & Interactive Telemetry Dashboard

### 5.1 Evolvable Code Region (`# EVOLVE-BLOCK`)
In `src/program.py`, only the `engineer_features(df)` function enclosed between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` is subject to mutation by Gemini; the surrounding LightGBM Tweedie training pipeline and evaluation interfaces remain frozen:

```python
# EVOLVE-BLOCK-START
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Candidate feature engineering logic evolves here...
    return df
# EVOLVE-BLOCK-END
```

### 5.2 3-Way Scientific Data Partitioning & Isolation
1. **Training Split (60%, 45,278 samples)**: Evolved candidate variants fit the LightGBM Tweedie regressor on this split.
2. **Evaluation Split (20%, 15,093 samples)**: Evaluates candidates during search to guide parent selection and crown the Champion.
3. **Holdout Split (20%, 15,093 samples)**: Strictly air-gapped throughout the entire evolutionary lifecycle; only evaluated once post-run to provide an unbiased comparison between Seed and Champion.

### 5.3 Interactive HTML Report Dashboard Features
The generated `artifacts/<task_id>/evolution_report.html` provides in-depth telemetry:
- **⚡ Compute & Resource Telemetry**: Displays total execution time, LLM invocation counts split by model (`gemini-3.5-flash` vs `gemini-3.1-pro-preview`), and verified Google Cloud API `inputTokenCount` and `outputTokenCount`.
- **👑 Champion Metric Prominence & Optimization Direction**:
  - Highlights the chosen optimization metric with an amber halo and ranking comparison.
  - The "Holdout Split Metric Comparison" table features a dedicated **Optimization Direction** column explicitly denoting whether each metric is `↑ Higher is better` or `↓ Lower is better`.
- **📈 Interactive Convergence Trajectory Chart (Hover Metric Cards)**:
  - Visualizes program generation lineage alongside the step-wise Pareto frontier envelope.
  - **Hover Interactivity**: Hovering over any scatter point smoothly magnifies the dot (normal points to 8px, 👑 Champion to 9.5px with gold halo) and renders a glassmorphic tooltip card following the cursor.
  - **Multi-Dimensional Metrics**: The tooltip presents the complete multi-metric evaluation profile:
    - `Normalized Gini (↑)`
    - `Top-10% Revenue Recall (↑)`
    - `RMSE Error (↓)` (restored to actual monetary dollar amounts)
    - `Spearman Rank Correlation (↑)`
  - Includes collision boundary detection to prevent screen overflow and fallback SVG `<title>` tags for print compatibility.
- **🔍 Code Diff Inspector**: Side-by-side color-coded diff of `# EVOLVE-BLOCK` between Seed Baseline and Champion.
- **🧠 Universal Evolution Attribution Engine**:
  - **Comprehensive Algorithmic Attribution**: Capable of attributing feature transformations, model hyperparameter configurations, custom loss functions, and sampling strategies.
  - **Domain-Infused Insights**: Anchored in mobile F2P monetization principles (180-day target, D0-D7 observation window, zero-inflation, whale dynamics) for mathematical and commercial rationale.
  - **Architectural Pruning & Strategic Takeaways**: Analyzes code pruning and synthesizes 3 industrial-grade deployment insights.
  - **Local Persistent Cache**: Attribution results persist to `code_attribution.json` for sub-50ms instant reload upon subsequent report opens.

---

## 6. Industrial Evaluation Metrics & Business Rationale

In 180-day mobile game LTV forecasting, four complementary metrics assess algorithmic robustness and commercial impact:

| Metric Name | Direction | Statistical / Algorithmic Formulation | Business Significance in Mobile Gaming |
| :--- | :---: | :--- | :--- |
| **Normalized Gini** | **`↑ Higher is better`**<br>(Range $`[0, 1]`$) | Measures the area under the Lorenz curve relative to an ideal ranking: $`\mathrm{Gini}(y, \hat{y}) / \mathrm{Gini}(y, y)`$. | **Core ranking metric for player segmentation and UA bid optimization**. Provides rank-order robustness against extreme outliers without distorting global cohort assessment. |
| **Top-10% Revenue Recall** | **`↑ Higher is better`**<br>(Percentage) | The percentage of total actual revenue contributed by players predicted in the top 10% highest LTV tier. | **Whale capture rate critical to game monetization**. Top 10% payers in F2P games typically drive 70%~85% of total revenue; directly impacts VIP operations and high-value cohort targeting. |
| **RMSE**<br>(Root Mean Squared Error) | **`↓ Lower is better`**<br>(Absolute USD) | Square root of the mean squared difference between predicted and actual revenue: $`\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}`$. | **Global revenue forecasting error**. Penalizes large dollar variances heavily due to Pareto skewness. *(Trained as `neg_rmse` in cloud optimizer for maximization; restored to positive currency in reports).* |
| **Spearman Rank Correlation** | **`↑ Higher is better`**<br>(Range $`[-1, 1]`$) | Pearson correlation between the ranked values of predictions and actuals: $`\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}`$. | **Pure monotonic rank ordering**. Completely invariant to non-linear monotonic shifts, reflecting algorithmic fidelity in correctly ordering player value. |

---

## 7. Project Delivery Sanitization & Privacy Protection (`make mask` & `make unmask`)

To safeguard enterprise privacy, execute the credential sanitization tool prior to open-sourcing, sharing with clients, or pushing to public repositories.

### 7.1 Masking Commands
```bash
# 1. Preview mode (inspect affected files without writing to disk)
make mask-dry

# 2. Execute sanitization (replaces project_id & ge_app_id with placeholders and saves local backup)
make mask
```

The sanitization utility automatically scans `config.yaml`, documentation, and scripts, replacing active credentials with `<YOUR_GCP_PROJECT_ID>` and `<YOUR_GE_APP_ID>` while injecting configuration guidance comments. Original credentials are safely backed up to `.credentials.backup` (protected by `.gitignore` and never committed to Git).

### 7.2 Credentials Restoration Commands
```bash
# 1. Automatic local restore (reads from .credentials.backup):
make unmask

# 2. Or restore with custom credentials:
make unmask PROJECT_ID=my-gcp-project-123 APP_ID=gemini-enterprise-999999
```

### 7.3 Clean Local Virtual Environment
```bash
rm -rf venv/
```
Recipients can initialize their isolated local environment in seconds by executing `make setup` (or running `python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`) as detailed in Section 2.
