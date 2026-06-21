"""Load the preprocessed dataset into a SQL table (AWS RDS MySQL).

Reads claim_approval_feature_dataset_v2.xlsx, drops the excluded columns
(`other`, `issueDesc`), and writes every remaining column/row to a SQL table
(default name: ``claim_dataset_v2``). Re-running replaces the table.

Run from the project root (DB_* must be set in .env):
    python src/load_dataset_to_db.py
    python src/load_dataset_to_db.py --table my_table_name
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import db
from config import DATA_PATH, EXCLUDE_COLS

DEFAULT_TABLE = "claim_dataset_v2"


def main(table: str = DEFAULT_TABLE, path=DATA_PATH) -> None:
    engine = db.get_engine()
    if engine is None:
        raise SystemExit(
            "Database is not configured (set DB_HOST/DB_NAME/DB_USER/DB_PASSWORD in .env).")

    df = pd.read_excel(path)
    dropped = [c for c in EXCLUDE_COLS if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"Source: {Path(path).name}")
    print(f"Dropped columns: {dropped or 'none'}")
    print(f"Loading {df.shape[0]} rows x {df.shape[1]} columns into table `{table}` ...")

    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=500)

    # Confirm what landed in the DB.
    from sqlalchemy import text
    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar()
        cols = [r[0] for r in conn.execute(
            text("SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                 "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t ORDER BY ORDINAL_POSITION"),
            {"t": table})]
    print(f"Done. Table `{table}` now has {n} rows and {len(cols)} columns:")
    print("  " + ", ".join(cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default=DEFAULT_TABLE, help="Target SQL table name")
    args = ap.parse_args()
    main(args.table)
