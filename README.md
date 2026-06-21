# Claim Approval — ML Optimization with MLflow + Optuna

An end-to-end, **explainable** machine-learning system that predicts insurance device-claim
`status` (**Declined** = positive class `1`, **Completed** = negative class `0`) and explains
*why* the winning model and hyperparameters were chosen.

It trains and Optuna-optimizes four model families (Random Forest, XGBoost, LightGBM, CatBoost),
tracks every experiment in MLflow, tunes the decision threshold for the imbalanced minority class,
and serves the result through a FastAPI backend and a React dashboard.

> **Headline result (this run):** the best model is **CatBoost (optimized)** — selected on
> validation **PR-AUC = 0.306**; test PR-AUC = 0.303, Declined **recall = 0.75**, balanced
> accuracy = 0.61 at the tuned threshold **0.44**. See `reports/final_model_report.md`.

---

## Why this design (the imbalance problem)

Declined claims are only ~16% of the data. A model that always predicts "Completed" would score
~84% accuracy while catching **zero** declines — so **accuracy is the wrong objective**. We
optimize and rank on **PR-AUC** (how well the rare class is ranked across all thresholds) and report
**Recall / Precision / F1 for Declined** and **Balanced Accuracy**. Accuracy is still shown, but
visually de-emphasized.

Principles enforced (project brief §14): stratified 70/15/15 split; preprocessing fit on **train
only**; the **test set is used exactly once**; no oversampling of val/test; reproducible seeds.

---

## Project structure

```
bolttech-prac/
├── data/preprocessed_data/claim_approval_feature_dataset_v2.xlsx
├── mlruns/                      # MLflow tracking store (generated)
├── models/                      # best_model.joblib + best_model_meta.json (generated)
├── reports/
│   ├── figures/                 # confusion / PR / ROC / importance per model (generated)
│   ├── model_comparison.csv     # leaderboard (generated)
│   ├── threshold_analysis.csv   # per-threshold metrics for best model (generated)
│   ├── final_model_report.md    # plain-English report (generated)
│   └── dashboard_data.json      # single bundle consumed by API + front-end (generated)
├── src/                         # Python backend (modules run with `--app-dir src`)
│   ├── config.py                # paths, constants, seed, positive-class convention, .env loader
│   ├── model_factory.py         # build/fit 4 families + ClaimModel (class of the saved model artifact)
│   ├── ml/                      # offline training/evaluation library (used by run_pipeline.py)
│   │   ├── data.py              # load, validate, encode target, stratified split
│   │   ├── preprocessing.py     # tree pipeline (impute+onehot) / CatBoost native cats
│   │   ├── train_baselines.py   # baseline configs per family
│   │   ├── optimize_optuna.py   # Optuna studies (maximize val PR-AUC) + nested MLflow runs
│   │   ├── evaluate.py          # metrics + plots
│   │   ├── threshold_tuning.py  # 0.05–0.95 sweep, pick best F1(Declined)
│   │   ├── explainability.py    # feature-importance grouping
│   │   └── mlflow_tracking.py   # MLflow helpers
│   ├── run_pipeline.py          # ORCHESTRATOR — trains/optimizes, writes all artifacts
│   ├── load_dataset_to_db.py    # load the dataset into a SQL table
│   ├── dashboard_api.py                 # FastAPI dashboard backend (uvicorn dashboard_api:app)
│   ├── prediction_service_api.py             # FastAPI production serving API (uvicorn prediction_service_api:app)
│   ├── db.py                    # RDS/MySQL persistence
│   ├── llm_explain.py           # LangChain + gpt-4o-mini explanations
│   └── prompts/                 # LLM prompt templates (per audience)
├── model-dashboard/             # Front-end app #1: model-optimization dashboard (Vite + React, 7 pages)
├── prediction-app/              # Front-end app #2: claim prediction & review (Vite + React, 5 tabs)
├── notebooks/model_experiment_summary.ipynb
├── docs/
│   ├── mlflow.md                # role of MLflow (experiment tracking)
│   ├── optuna.md                # role of Optuna (hyperparameter search)
│   ├── hyperparameter_importance.md  # how hyperparameter importance is computed
│   └── feature_importance.md         # how feature importance is computed
├── pyproject.toml               # project + dependencies (uv)
├── uv.lock                      # pinned, reproducible dependency lock
├── Dockerfile                   # multi-stage build for the serving API
├── .dockerignore
└── README.md
```

## Documentation
- [`docs/optuna.md`](docs/optuna.md) — how Optuna searches hyperparameters (objective = validation PR-AUC, search spaces, reproducibility, trial count).
- [`docs/mlflow.md`](docs/mlflow.md) — how MLflow tracks every model and trial (run hierarchy, what's logged, how to view).
- [`docs/hyperparameter_importance.md`](docs/hyperparameter_importance.md) — how hyperparameter importance is computed (Optuna fANOVA over the search).
- [`docs/feature_importance.md`](docs/feature_importance.md) — how feature importance is computed (per-family model-internal importance + one-hot grouping).

---

## 1. Install dependencies

Python 3.11+ (developed on 3.14). Dependencies are managed with [uv](https://docs.astral.sh/uv/)
(`pyproject.toml` + `uv.lock`). LightGBM needs the OpenMP runtime.

```bash
# system lib for LightGBM (Debian/Ubuntu)
sudo apt-get install -y libgomp1

# install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                 # creates .venv and installs the locked dependencies
```

Then either activate the env (`source .venv/bin/activate`) or prefix commands with `uv run`
(e.g. `uv run python src/run_pipeline.py`, `uv run uvicorn prediction_service_api:app --app-dir src --port 8001`).

Front-end (Node 18+) — there are two independent apps; install whichever you'll run:

```bash
cd model-dashboard && npm install   # app #1: model-optimization dashboard
cd prediction-app  && npm install   # app #2: claim prediction & review
```

## 2. Train + optimize everything

```bash
python src/run_pipeline.py            # default 30 Optuna trials per model
N_TRIALS=60 python src/run_pipeline.py   # more trials = better search (slower)
```

This single command: loads & validates data → trains 4 baselines → runs Optuna for each family →
tunes the threshold on validation → evaluates the best of each on test → logs everything to MLflow →
selects the overall best by validation PR-AUC → saves the model and **all** reports + figures +
`dashboard_data.json`. Runtime: a few minutes on 2 cores.

## 3. View MLflow (experiment tracking)

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
# open http://localhost:5000
```

Structure: one **parent run per family** (`*_optuna`) with one **nested child run per Optuna trial**
(params + `val_pr_auc`), plus a `*_baseline` and `*_optimized` evaluation run per family that logs
hyperparameters, the imbalance strategy, validation + test metrics, the chosen threshold, confusion
/ PR / ROC / feature-importance images and the model artifact.

## 4. Model comparison results

- `reports/model_comparison.csv` — every baseline + optimized model with val/test metrics, FP/FN.
- `reports/final_model_report.md` — the comparison table + a plain-English selection rationale.
- `reports/threshold_analysis.csv` — per-threshold precision/recall/F1/balanced-accuracy.
- `notebooks/model_experiment_summary.ipynb` — runnable summary (table, figures, recommendation).

## 5. Start the FastAPI server

```bash
source .venv/bin/activate
uvicorn dashboard_api:app --app-dir src --reload --port 8000
# docs at http://localhost:8000/docs
```

Endpoints: `POST /predict`, `GET /model-summary`, `GET /model-comparison`,
`GET /threshold-analysis`, `GET /feature-importance`, `GET /explain`, `GET /dashboard`.

`GET /explain` returns an LLM-generated (LangChain + `gpt-4o-mini`) plain-English explanation of the
model and its feature importance. It requires an OpenAI key in the server environment:

```bash
OPENAI_API_KEY=sk-... uv run uvicorn dashboard_api:app --app-dir src --port 8000
```

Without the key, `/explain` returns `503` with a clear message (other endpoints are unaffected).

### Saving predictions to AWS RDS (MySQL)

If the `DB_*` variables are set (see `.env.example`), every `POST /predict` saves the input features
and the prediction to the `claim_predictions` table in MySQL — one row per prediction, **one column
per feature** (numeric → `DOUBLE`, categorical → `VARCHAR`) plus `predicted_class`, `predicted_label`,
`probability_declined/completed`, `threshold_used`, `model_version`, `id`, `created_at`. The table is
**auto-created** on first save (`CREATE TABLE IF NOT EXISTS`, columns derived from the model).

```bash
# in .env
DB_HOST=<your-instance>.rds.amazonaws.com
DB_PORT=3306
DB_NAME=bolttech_prac
DB_USER=admin
DB_PASSWORD=********
```

Saving is **best-effort**: if the DB is unset or unreachable, `/predict` still returns the prediction
(the response includes `"saved_to_db": true|false`). The EC2/host must be able to reach RDS on 3306
(RDS security group + VPC). Uses SQLAlchemy + PyMySQL.

View saved rows via **`GET /predictions/recent?limit=25`** (newest first; `enabled:false` when the DB
isn't configured), or the **History** tab in the prediction-app.

LLM explanations can be attached to a saved row: `/predict` returns the `prediction_id`, and
**`POST /predictions/{id}/explanation`** `{kind: "adjuster"|"customer", explanation}` stores the text
in the `adjuster_explanation` / `customer_explanation` columns (added automatically to the table).
The Claims Adjuster and Customer tabs have a "Save explanation to database" button, and the History
table shows both columns.

The Claims Adjuster tab also has **Approve → Completed** / **Decline → Declined** buttons that record
the adjuster's final human decision via **`POST /predictions/{id}/decision`** `{decision}` into the
`adjuster_decision` column (the model's `predicted_class` is preserved, so an override is visible).
The History table shows both "Result (model)" and "Adjuster decision".

To load the full preprocessed dataset into its own SQL table (everything in
`claim_approval_feature_dataset_v2.xlsx` except `other`/`issueDesc` — 2,880 rows × 26 columns):

```bash
python src/load_dataset_to_db.py            # table: claim_dataset_v2 (re-run replaces it)
python src/load_dataset_to_db.py --table my_name
```

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"features": {"rrp": 1799, "excessFee": 139, "coverage": "ADLD",
       "deviceType": "SMARTPHONES", "country": "SE", "claimType": "Accidental Damage"}}'
```

Unspecified features are imputed automatically. The decision uses the tuned threshold unless you
pass `"threshold"` in the request.

### Production serving API (`prediction_service_api.py`)

A second, ops-focused inference service (separate from `dashboard_api.py`, which backs the dashboard) with
health/readiness probes, batch inference, and service metrics:

```bash
uvicorn prediction_service_api:app --app-dir src --port 8001
# docs at http://localhost:8001/docs
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness — process is up. |
| `GET /ready` | Readiness — model loaded and usable (503 otherwise). |
| `GET /metadata` | Model version, training date, hyperparameters, test metrics, feature importance. |
| `POST /predict` | Single-sample inference (validated). |
| `POST /predict-batch` | Batch inference for `samples: [...]` (1–1000). |
| `GET /metrics` | Request count, error rate, latency (avg/p50/p95/p99), per-path counts, uptime. |

```bash
curl -X POST localhost:8001/predict -H 'Content-Type: application/json' \
  -d '{"features": {"rrp": 1799, "coverage": "ADLD", "claimType": "Theft"}, "threshold": 0.5}'

curl -X POST localhost:8001/predict-batch -H 'Content-Type: application/json' \
  -d '{"samples": [{"rrp": 1799, "coverage": "ADLD"}, {"rrp": 40000, "claimType": "Liquid Damage"}]}'
```

Validation returns `422` with a structured `{"error": {...}}` body for unknown feature names,
wrong numeric types, out-of-range thresholds, or empty batches. If the model artifact is missing,
`/ready`, `/metadata`, `/predict*` return `503` (run `python src/run_pipeline.py` first).

## 6. Start the front-end apps

This repo has **two independent front-end apps** (each with its own `README.md`):

| App | Folder | Purpose | Dev port |
|---|---|---|---|
| **Model-optimization dashboard** | [`model-dashboard/`](model-dashboard/) | Visualize/compare the model optimization & selection (analytics audience). | 5173 |
| **Claim prediction & review** | [`prediction-app/`](prediction-app/) | Score a claim, get adjuster/customer explanations, record the adjuster decision, view history. | 5174 |

```bash
cd model-dashboard
npm run dev          # http://localhost:5173  (proxies /api -> :8000 for the demo)
# or a static build:
npm run build && npm run preview
```

The dashboard loads `public/dashboard_data.json` (copied automatically by the pipeline), so pages
A–F work **without** the backend. Only the **Prediction Demo** page needs the FastAPI server running
(it calls `/api/predict`, proxied to `:8000`). For the second app see [`prediction-app/README.md`](prediction-app/README.md).

### Dashboard pages (model-dashboard)
- **Overview** — dataset size, class distribution (bar + donut), imbalance ratio, split sizes.
- **Model Leaderboard** — sortable table of all 8 models; best row highlighted; Accuracy de-emphasized.
- **MLflow / Optuna** — per-family search history, running-best curve, parameter importance, best trial.
- **Metric Explanation** — business-friendly definitions of PR-AUC, Recall, Precision, F1, etc.
- **Threshold Tuning** — precision/recall/F1 vs threshold, PR curve, confusion matrix at the selection.
- **Final Model** — selected model, hyperparameters, test metrics, feature importance, rationale.
- **Prediction Demo** — enter feature values → class + P(Declined)/P(Completed) + threshold used.

## 7. Docker (serving API)

A multi-stage `Dockerfile` builds the serving API: dependencies are installed from `uv.lock`
(reproducible) into a slim Python 3.12 image, with `libgomp1` for LightGBM and the trained model
baked in.

```bash
docker build -t bolttech-prac .
docker run --rm -p 8000:8000 bolttech-prac      # API + docs at http://localhost:8000/docs
```

The default command runs `prediction_service_api:app`. Override it to run the dashboard backend or retrain:

```bash
docker run --rm -p 8000:8000 bolttech-prac uvicorn dashboard_api:app --app-dir src --host 0.0.0.0 --port 8000
docker run --rm bolttech-prac python src/run_pipeline.py
```

The image includes a `HEALTHCHECK` against `/health`. (The React front-end is a separate app and is
not part of this image — see `.dockerignore`.)

---

## How to read the metrics (quick reference)

| Metric | Plain English |
|---|---|
| **PR-AUC** | How well the model separates the rare *Declined* class across all thresholds. **Primary metric.** |
| **Recall (Declined)** | Of truly declined claims, how many did we catch? |
| **Precision (Declined)** | Of claims we flagged as declined, how many really were? |
| **F1 (Declined)** | Balance of the two above (used to pick the threshold). |
| **Balanced Accuracy** | Average recall over both classes — fair despite imbalance. |
| **ROC-AUC** | Overall ranking quality; can look optimistic under imbalance. |
| **Accuracy** | Share correct — *misleading here* (always-Completed ≈ 84%). De-emphasized. |

## Reproducibility & notes
- Seed `42` throughout (`config.RANDOM_STATE`), `TPESampler(seed=42)` for Optuna.
- The MLflow **file store** is enabled via `MLFLOW_ALLOW_FILE_STORE=true` (set automatically). For
  the MLflow **model registry** you would point MLflow at a database backend (e.g.
  `sqlite:///mlflow.db`); the best model is also saved directly to `models/best_model.joblib`.
- `mlruns/`, `model-dashboard/node_modules/`, `prediction-app/node_modules/` and the apps' `dist/` are git-ignored (regenerate them).
```text
Best Model: CatBoost (optimized)   ·   Strategy: class_weights {Completed:1, Declined:5.36}
Best Threshold: 0.44   ·   Selected on highest validation PR-AUC
```
