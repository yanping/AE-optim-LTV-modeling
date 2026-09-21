# ==============================================================================
# Makefile for AlphaEvolve LTV Feature Engineering Optimization
# ==============================================================================

SHELL := /bin/bash
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help setup auth split run report test mask mask-dry clean

# Default goal
help:
	@echo "Available commands:"
	@echo "  make setup            - Initialize Python virtual environment (venv) and install dependencies"
	@echo "  make auth             - Authenticate with Google Cloud ADC (gcloud auth application-default login)"
	@echo "  make split            - Split data/train_wide.csv into 60% Train, 20% Eval, 20% Holdout"
	@echo "  make run              - Run AlphaEvolve evolution (options: programs=N, metric=gini|recall|rmse|spearman, task_id=ID)"
	@echo "  make report           - Open HTML report in default browser (options: task_id=ID, default: latest task)"
	@echo "  make test             - Run automated tests in tests/"
	@echo "  make mask             - Sanitize & mask personal project_id/ge_app_id before project delivery"
	@echo "  make mask-dry         - Preview masked credentials without modifying files"
	@echo "  make clean            - Remove cache files and build artifacts"

setup:
	@echo "Setting up virtual environment in $(VENV)..."
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv $(VENV); \
	fi
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "Setup completed successfully."

auth:
	@echo "Authenticating Google Cloud Application Default Credentials (ADC)..."
	@gcloud auth application-default login

split:
	@echo "Splitting dataset into 60% Train, 20% Eval, and 20% Holdout..."
	@$(PYTHON) -c "from src.evaluate import get_cached_splits; get_cached_splits()"

run:
	@echo "Launching AlphaEvolve evolution..."
	@$(PYTHON) -m src.run_evolution $(if $(programs),--programs $(programs),) $(if $(metric),--metric $(metric),) $(if $(task_id),--task-id $(task_id),)

report:
	@echo "Opening HTML evolution summary report in default browser..."
	@$(PYTHON) -m src.report $(if $(task_id),--task-id $(task_id),) --open

test:
	@echo "Running tests..."
	@$(PYTHON) -m pytest tests/ -v

mask:
	@python3 scripts/mask_credentials.py

mask-dry:
	@python3 scripts/mask_credentials.py --dry-run

unmask:
	@python3 scripts/mask_credentials.py --restore $(if $(PROJECT_ID),--project-id $(PROJECT_ID),) $(if $(APP_ID),--app-id $(APP_ID),)

clean:
	@echo "Cleaning temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .coverage htmlcov 2>/dev/null || true
	@echo "Clean completed."
