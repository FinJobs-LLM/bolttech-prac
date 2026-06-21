"""Central configuration: paths, constants, and shared settings.

Everything that other modules need to agree on (file locations, the target
column, the positive-class convention, random seed) lives here so the pipeline
stays consistent and reproducible.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# Load environment variables from a local .env (e.g. OPENAI_API_KEY) if present.
# Existing environment variables take precedence (override=False).
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:  # python-dotenv is optional; env vars still work without it
    pass

DATA_PATH = ROOT / "data" / "preprocessed_data" / "claim_approval_feature_dataset_v2.xlsx"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MLRUNS_DIR = ROOT / "mlruns"

# Machine-readable bundle the API and the front-end both consume.
DASHBOARD_JSON = REPORTS_DIR / "dashboard_data.json"
COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
THRESHOLD_CSV = REPORTS_DIR / "threshold_analysis.csv"
FINAL_REPORT_MD = REPORTS_DIR / "final_model_report.md"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
BEST_MODEL_META = MODELS_DIR / "best_model_meta.json"

# --- Modeling constants --------------------------------------------------
TARGET = "status"
POSITIVE_LABEL = "Declined"   # encoded as 1 (the minority class we care about)
NEGATIVE_LABEL = "Completed"  # encoded as 0
EXCLUDE_COLS = ["other", "issueDesc"]

# Recommended imbalance weight from the project brief (~ majority/minority).
IMBALANCE_WEIGHT = 5.36

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the *whole* dataset

# Threshold search grid (inclusive of both ends, step 0.01).
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95

# Optuna trial counts (override with env var for quick smoke runs).
N_TRIALS = int(os.environ.get("N_TRIALS", "50"))

MLFLOW_EXPERIMENT = "claim_approval_optimization"

ALL_MODELS = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]

# Deployed release identifier — set to the git tag by cd.yml at image build time
# (ARG/ENV APP_VERSION). Lets a running service report which release it is.
APP_VERSION = os.environ.get("APP_VERSION", "dev")


def ensure_dirs() -> None:
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR, MLRUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
