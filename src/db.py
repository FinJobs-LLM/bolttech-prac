"""Persistence of predictions to AWS RDS (MySQL).

When the DB_* environment variables are configured (loaded from .env by
``config``), each prediction made by ``/predict`` is saved to the
``claim_predictions`` table — one row per prediction, with one column per input
feature plus the prediction outputs.

Design notes:
* The table is created on demand (``CREATE TABLE IF NOT EXISTS``), with columns
  derived from the served model's own feature lists, so it always matches the
  model (numeric features -> DOUBLE, categorical -> VARCHAR).
* Saving is best-effort: failures are logged and swallowed so a database problem
  never breaks prediction serving.
"""
from __future__ import annotations

import logging
import os

from sqlalchemy import text

# config loads .env (DB_* etc.) on import.
import config  # noqa: F401

log = logging.getLogger("claim_db")

TABLE = "claim_predictions"
_engine = None          # lazily created SQLAlchemy engine
_table_ready = False     # CREATE TABLE IF NOT EXISTS run once per process

# Prediction columns (name -> MySQL type), in insert order.
_PRED_COLS = {
    "predicted_class": "VARCHAR(16)",
    "predicted_label": "INT",
    "probability_declined": "DOUBLE",
    "probability_completed": "DOUBLE",
    "threshold_used": "DOUBLE",
    "model_version": "VARCHAR(128)",
}

# Free-text columns filled in later (via /predictions/{id}/explanation), one per
# explanation audience. Added to existing tables by _ensure_explanation_columns.
_EXPLANATION_COLS = {
    "adjuster": "adjuster_explanation",
    "customer": "customer_explanation",
}
_explain_cols_ready = False


def db_enabled() -> bool:
    """True if the minimum DB connection settings are present."""
    return all(os.environ.get(k) for k in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"))


def _get_engine():
    global _engine
    if _engine is None:
        if not db_enabled():
            return None
        from sqlalchemy import create_engine

        host = os.environ["DB_HOST"]
        port = os.environ.get("DB_PORT", "3306")
        name = os.environ["DB_NAME"]
        user = os.environ["DB_USER"]
        pwd = os.environ["DB_PASSWORD"]
        url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{name}?charset=utf8mb4"
        # pool_pre_ping recovers from dropped connections; pool_recycle avoids
        # MySQL's idle timeout closing pooled connections.
        _engine = create_engine(url, pool_pre_ping=True, pool_recycle=1800, future=True)
    return _engine


def get_engine():
    """Public accessor for the configured SQLAlchemy engine (or None if the DB
    is not configured). Used by scripts such as load_dataset_to_db.py."""
    return _get_engine()


def _feature_columns(model) -> dict[str, str]:
    """name -> MySQL type for each model feature (numeric DOUBLE, else VARCHAR)."""
    num = set(model.num_cols)
    cols = {}
    for f in model.feature_cols:
        cols[f] = "DOUBLE" if f in num else "VARCHAR(255)"
    return cols


def ensure_table(model) -> None:
    """Create the predictions table if it does not exist (idempotent)."""
    global _table_ready
    if _table_ready:
        return
    engine = _get_engine()
    if engine is None:
        return
    cols = []
    for name, typ in _feature_columns(model).items():
        cols.append(f"`{name}` {typ} NULL")
    for name, typ in _PRED_COLS.items():
        cols.append(f"`{name}` {typ} NULL")
    for col in _EXPLANATION_COLS.values():
        cols.append(f"`{col}` TEXT NULL")
    ddl = (
        f"CREATE TABLE IF NOT EXISTS `{TABLE}` (\n"
        "  `id` BIGINT AUTO_INCREMENT PRIMARY KEY,\n"
        "  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,\n"
        + ",\n".join("  " + c for c in cols)
        + "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    )
    with engine.begin() as conn:
        conn.execute(text(ddl))
    _ensure_explanation_columns(engine)  # add cols to pre-existing tables
    _table_ready = True


def _ensure_explanation_columns(engine) -> None:
    """Add the explanation TEXT columns to an existing table if missing.

    MySQL has no ``ADD COLUMN IF NOT EXISTS``, so we check information_schema.
    Idempotent; runs at most once per process."""
    global _explain_cols_ready
    if _explain_cols_ready:
        return
    with engine.begin() as conn:
        existing = {
            r[0] for r in conn.execute(
                text("SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                     "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t"),
                {"t": TABLE},
            )
        }
        for col in _EXPLANATION_COLS.values():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE `{TABLE}` ADD COLUMN `{col}` TEXT NULL"))
    _explain_cols_ready = True


def save_prediction(model, feature_row: dict, prediction: dict):
    """Insert one prediction row. Best-effort: returns the new row's id on
    success, or None on any failure (the error is logged, never raised)."""
    try:
        engine = _get_engine()
        if engine is None:
            return None
        ensure_table(model)

        feat_cols = list(model.feature_cols)
        params = {f: feature_row.get(f, None) for f in feat_cols}
        for k in _PRED_COLS:
            params[k] = prediction.get(k)

        all_cols = feat_cols + list(_PRED_COLS.keys())
        col_sql = ", ".join(f"`{c}`" for c in all_cols)
        val_sql = ", ".join(f":{c}" for c in all_cols)
        insert = text(f"INSERT INTO `{TABLE}` ({col_sql}) VALUES ({val_sql})")
        with engine.begin() as conn:
            result = conn.execute(insert, params)
            return int(result.lastrowid)
    except Exception as exc:  # best-effort: never break prediction serving
        log.warning("Failed to save prediction to DB: %s: %s", type(exc).__name__, exc)
        return None


def save_explanation(prediction_id: int, kind: str, explanation: str) -> dict:
    """Save an LLM explanation onto an existing prediction row.

    `kind` is 'adjuster' or 'customer'. Returns a dict:
    {"ok": bool, "reason": None | 'db_disabled' | 'bad_kind' | 'not_found' | 'error: ...'}.
    """
    if kind not in _EXPLANATION_COLS:
        return {"ok": False, "reason": "bad_kind"}
    engine = _get_engine()
    if engine is None:
        return {"ok": False, "reason": "db_disabled"}
    col = _EXPLANATION_COLS[kind]
    try:
        _ensure_explanation_columns(engine)
        with engine.begin() as conn:
            res = conn.execute(
                text(f"UPDATE `{TABLE}` SET `{col}` = :e WHERE `id` = :id"),
                {"e": explanation, "id": int(prediction_id)},
            )
        return {"ok": res.rowcount > 0, "reason": None if res.rowcount > 0 else "not_found"}
    except Exception as exc:
        log.warning("save_explanation failed: %s: %s", type(exc).__name__, exc)
        return {"ok": False, "reason": f"error: {exc}"}


def recent_predictions(limit: int = 20):
    """Return the most recent saved predictions (newest first).

    Returns None if the DB is not configured, or [] on any read error / if the
    table does not exist yet.
    """
    engine = _get_engine()
    if engine is None:
        return None
    try:
        limit = max(1, min(int(limit), 500))
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM `{TABLE}` ORDER BY `id` DESC LIMIT {limit}")
            ).mappings().all()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.warning("recent_predictions failed: %s: %s", type(exc).__name__, exc)
        return []
