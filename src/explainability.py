"""Feature-importance extraction and grouping for explainability.

One-hot encoding explodes a single categorical column into many binary columns.
For business-facing explanations we also roll those back up to the original
feature so a reader sees "retailerName" rather than 30 "retailerName_*" bars.
"""
from __future__ import annotations

import re

from model_factory import ClaimModel


def raw_importances(model: ClaimModel) -> dict[str, float]:
    """Importance per (possibly one-hot) model input feature."""
    return model.feature_importance()


def grouped_importances(model: ClaimModel) -> dict[str, float]:
    """Importance summed back to the original feature names.

    One-hot names look like ``<col>_<value>``; we attribute each back to
    ``<col>`` when the prefix matches an original categorical column.
    """
    raw = model.feature_importance()
    cat_cols = sorted(model.cat_cols, key=len, reverse=True)
    grouped: dict[str, float] = {c: 0.0 for c in model.feature_cols}
    for name, val in raw.items():
        matched = None
        for c in cat_cols:
            if name == c or name.startswith(c + "_"):
                matched = c
                break
        if matched is None:
            # numeric feature (name is the column itself) or unknown -> keep as-is
            matched = name if name in grouped else name
            grouped.setdefault(matched, 0.0)
        grouped[matched] = grouped.get(matched, 0.0) + float(val)
    # Drop zero-importance features for a cleaner view, keep sorted desc.
    return dict(sorted(grouped.items(), key=lambda kv: kv[1], reverse=True))
