# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This project predicts insurance device-claim approval (`status`: Declined=1 positive, Completed=0).
It now contains a full ML system — see `README.md` for the authoritative guide:
- `src/` — training pipeline (`run_pipeline.py` orchestrates), Optuna optimization, MLflow tracking,
  threshold tuning, evaluation, explainability, and a FastAPI service (`serve.py`).
- `model-dashboard/` — Vite + React model-optimization explainability dashboard (front-end app #1).
- `prediction-app/` — Vite + React claim prediction & review app (front-end app #2).
- `reports/`, `models/`, `mlruns/` — generated artifacts (run `python src/run_pipeline.py`).
- `pyproject.toml` + `uv.lock` — Python deps managed with `uv` (`uv sync`); LightGBM needs `libgomp1` (`apt-get install -y libgomp1`).

Modeling rules (do not break): optimize/rank on **PR-AUC**, not accuracy; stratified 70/15/15; fit
preprocessing on train only; the test set is used once; seed 42 everywhere.

## Layout

- `data/original_data/` — empty; intended for raw, untransformed source data.
- `data/preprocessed_data/claim_approval_feature_dataset.xlsx` — the working dataset (single sheet `Sheet1`, ~2880 rows + header, 28 columns).

Keep this split: raw inputs in `original_data/`, model-ready features in `preprocessed_data/`.

## Dataset: `claim_approval_feature_dataset.xlsx`

One row per insurance claim. **Prediction target is `status`** (`Completed` ≈ approved, `Declined`) — note the class imbalance (~2427 Completed vs ~453 Declined), so account for it in any modeling/metrics.

Column groups:
- **Target:** `status`.
- **Free text:** `issueDesc` — the claimant's incident narrative, in mixed languages (Swedish, Dutch, etc.). **PII is masked with `*****` runs**; preserve this masking and treat the field as multilingual. Text encoding in the raw cells is mojibake (e.g. `Ã¥` for `å`) — decode/normalize before NLP use.
- **Value / RRP features:** `excessFee`, `rrp`, `oldBalanceRRP`, `remaining_rrp_ratio`, `used_rrp_amount`, `has_prior_rrp_usage`.
- **Policy features:** `coverage` (`ADLD`, `ADLD/THEFT`), `policy_duration_months`, `policy_start_year`, `policy_start_month`, `days_from_purchase_to_policy`.
- **Claim/context categoricals:** `retailerName`, `deviceType` (mostly `SMARTPHONES`), `channel` (mostly `Online Portal`), `claimType` (`Accidental Damage`/`Theft`/`Liquid Damage`), `country` (`NL`/`SE`/`FI`).
- **Damage flags (mostly empty / 0/1):** `turnOnOff`, `touchScreen`, `smashed`, `frontCamera`, `backCamera`, `frontOrBackCamera`, `audio`, `mic`, `buttons`, `other`. These are sparsely populated — most rows leave them blank rather than `0`, so distinguish "blank" from "0" when imputing.
