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
├── src/
│   ├── config.py                # paths, constants, seed, positive-class convention
│   ├── data.py                  # load, validate, encode target, stratified split
│   ├── preprocessing.py         # tree pipeline (impute+onehot) / CatBoost native cats
│   ├── model_factory.py         # build/fit 4 families + ClaimModel wrapper (predict/importance)
│   ├── train_baselines.py       # baseline configs per family
│   ├── optimize_optuna.py       # Optuna studies (maximize val PR-AUC) + nested MLflow runs
│   ├── evaluate.py              # metrics + plots
│   ├── threshold_tuning.py      # 0.05–0.95 sweep, pick best F1(Declined)
│   ├── explainability.py        # feature-importance grouping
│   ├── mlflow_tracking.py       # MLflow helpers
│   ├── run_pipeline.py          # ORCHESTRATOR — runs everything, writes all artifacts
│   └── serve.py                 # FastAPI service
├── frontend/                    # Vite + React dashboard (7 pages)
├── notebooks/model_experiment_summary.ipynb
├── requirements.txt
└── README.md
```

---

## 1. Install dependencies

Python 3.11+ (developed on 3.14). LightGBM needs the OpenMP runtime.

```bash
# system lib for LightGBM (Debian/Ubuntu)
sudo apt-get install -y libgomp1

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Front-end (Node 18+):

```bash
cd frontend && npm install
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
uvicorn serve:app --app-dir src --reload --port 8000
# docs at http://localhost:8000/docs
```

Endpoints: `POST /predict`, `GET /model-summary`, `GET /model-comparison`,
`GET /threshold-analysis`, `GET /feature-importance`, `GET /dashboard`.

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"features": {"rrp": 1799, "excessFee": 139, "coverage": "ADLD",
       "deviceType": "SMARTPHONES", "country": "SE", "claimType": "Accidental Damage"}}'
```

Unspecified features are imputed automatically. The decision uses the tuned threshold unless you
pass `"threshold"` in the request.

## 6. Start the React front-end

```bash
cd frontend
npm run dev          # http://localhost:5173  (proxies /api -> :8000 for the demo)
# or a static build:
npm run build && npm run preview
```

The dashboard loads `public/dashboard_data.json` (copied automatically by the pipeline), so pages
A–F work **without** the backend. Only the **Prediction Demo** page needs the FastAPI server running
(it calls `/api/predict`, proxied to `:8000`).

### Dashboard pages
- **Overview** — dataset size, class distribution (bar + donut), imbalance ratio, split sizes.
- **Model Leaderboard** — sortable table of all 8 models; best row highlighted; Accuracy de-emphasized.
- **MLflow / Optuna** — per-family search history, running-best curve, parameter importance, best trial.
- **Metric Explanation** — business-friendly definitions of PR-AUC, Recall, Precision, F1, etc.
- **Threshold Tuning** — precision/recall/F1 vs threshold, PR curve, confusion matrix at the selection.
- **Final Model** — selected model, hyperparameters, test metrics, feature importance, rationale.
- **Prediction Demo** — enter feature values → class + P(Declined)/P(Completed) + threshold used.

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
- `mlruns/`, `frontend/node_modules/` and `frontend/dist/` are git-ignored (regenerate them).
```text
Best Model: CatBoost (optimized)   ·   Strategy: class_weights {Completed:1, Declined:5.36}
Best Threshold: 0.44   ·   Selected on highest validation PR-AUC
```
