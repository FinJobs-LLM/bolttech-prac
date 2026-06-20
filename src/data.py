"""Data loading, validation, target encoding and stratified splitting.

All splitting happens on the raw (un-preprocessed) frame so that preprocessing
can be fit on the training set only -- this is what prevents data leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    EXCLUDE_COLS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)


@dataclass
class Dataset:
    """Container for the fully prepared, split dataset."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    num_cols: list[str]
    cat_cols: list[str]
    feature_cols: list[str]
    class_counts: dict[str, int] = field(default_factory=dict)


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_excel(path)


def infer_feature_types(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Numeric dtype -> numerical feature; object/bool/category -> categorical.

    In the v2 dataset the device-damage flags contain the string ``"Unknown"``
    so they arrive as ``object`` columns and are correctly treated as
    categorical here.
    """
    num_cols, cat_cols = [], []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def validate_and_report(df: pd.DataFrame) -> dict:
    """Print a validation report and return a structured summary dict."""
    report: dict = {}
    report["shape"] = list(df.shape)
    report["columns"] = list(df.columns)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")

    vc = df[TARGET].value_counts(dropna=False)
    report["target_distribution"] = {str(k): int(v) for k, v in vc.items()}

    missing = df.isna().sum()
    report["missing_by_column"] = {c: int(missing[c]) for c in df.columns if missing[c] > 0}

    excluded_present = [c for c in EXCLUDE_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c != TARGET and c not in EXCLUDE_COLS]
    num_cols, cat_cols = infer_feature_types(df, feature_cols)

    report["excluded_columns"] = excluded_present
    report["numerical_features"] = num_cols
    report["categorical_features"] = cat_cols

    print("=" * 70)
    print("DATA VALIDATION REPORT")
    print("=" * 70)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Target column: '{TARGET}'")
    print(f"Target distribution: {report['target_distribution']}")
    print(f"Excluded columns present: {excluded_present or 'none'}")
    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    miss = report["missing_by_column"]
    print(f"Columns with missing values: {miss or 'none'}")
    print("=" * 70)
    return report


def encode_target(y: pd.Series) -> pd.Series:
    """Completed -> 0, Declined -> 1 (Declined is the positive class)."""
    mapping = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}
    unknown = set(y.dropna().unique()) - set(mapping)
    if unknown:
        raise ValueError(f"Unexpected target values: {unknown}")
    return y.map(mapping).astype(int)


def prepare_dataset(path=DATA_PATH) -> tuple[Dataset, dict]:
    """Load, validate, encode the target and produce stratified 70/15/15 splits."""
    df = load_raw(path)
    report = validate_and_report(df)

    feature_cols = [c for c in df.columns if c != TARGET and c not in EXCLUDE_COLS]
    num_cols, cat_cols = infer_feature_types(df, feature_cols)

    X = df[feature_cols].copy()
    y = encode_target(df[TARGET])

    # Step 1: carve off the test set (15% of all rows), stratified.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # Step 2: split the remaining 85% into train and validation so that
    # validation is VAL_SIZE of the *whole* dataset.
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_fraction, stratify=y_tmp, random_state=RANDOM_STATE
    )

    class_counts = {NEGATIVE_LABEL: int((y == 0).sum()), POSITIVE_LABEL: int((y == 1).sum())}

    print("\nSPLIT SIZES (stratified):")
    for name, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
        pos = int(ys.sum())
        print(f"  {name:5}: {len(ys):4} rows | Declined={pos} ({100*pos/len(ys):.1f}%)")

    ds = Dataset(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        num_cols=num_cols, cat_cols=cat_cols, feature_cols=feature_cols,
        class_counts=class_counts,
    )
    report["split_sizes"] = {
        "train": len(y_train), "val": len(y_val), "test": len(y_test),
    }
    report["class_counts"] = class_counts
    report["imbalance_ratio"] = round(class_counts[NEGATIVE_LABEL] / max(class_counts[POSITIVE_LABEL], 1), 2)
    return ds, report


if __name__ == "__main__":
    ds, rep = prepare_dataset()
    print("\nImbalance ratio (Completed:Declined):", rep["imbalance_ratio"])
