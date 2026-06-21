"""Preprocessing pipelines for the four model families.

* Tree models that need numeric input (RandomForest, XGBoost, LightGBM):
  median-impute numerics, impute categoricals with ``"Unknown"``, cast them to
  string and one-hot encode.
* CatBoost: keep the raw frame, cast categoricals to string (no NaN) and let
  CatBoost handle them natively via ``cat_features``.

All transformers are *fit on the training set only*; the same fitted objects are
applied to validation and test data, which avoids leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def _to_str(arr):
    """Cast every cell to string so OneHotEncoder sees consistent types."""
    return np.asarray(arr, dtype=object).astype(str)


def build_tree_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """ColumnTransformer for RandomForest / XGBoost / LightGBM.

    Scaling is intentionally omitted -- tree-based models are invariant to
    monotonic rescaling of individual features, so it adds nothing.
    """
    numeric = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])

    categorical = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("to_str", FunctionTransformer(_to_str, feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_catboost_frame(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Return a copy where categorical columns are NaN-free strings.

    CatBoost handles missing *numeric* values natively, but categorical
    features must be strings without NaN.
    """
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), "Unknown").astype(str)
    return X


def catboost_cat_indices(feature_cols: list[str], cat_cols: list[str]) -> list[int]:
    return [feature_cols.index(c) for c in cat_cols]
