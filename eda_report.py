"""Generate a feature-level EDA Markdown report for the claim-approval dataset.

Reads the Excel dataset, analyzes the specified numerical and categorical
features against the target column `status`, and writes a structured Markdown
report. Missing columns are reported in the output instead of failing.
"""

import os
import pandas as pd

# --- Configuration -------------------------------------------------------
DATA_PATH = "data/preprocessed_data/claim_approval_feature_dataset.xlsx"
OUTPUT_PATH = "data/preprocessed_data/claim_approval_feature_dataset_EDA.md"
TARGET = "status"

NUM_COLS = [
    "excessFee",
    "rrp",
    "oldBalanceRRP",
    "remaining_rrp_ratio",
    "used_rrp_amount",
    "policy_duration_months",
    "policy_start_year",
    "policy_start_month",
    "days_from_purchase_to_policy",
]

CAT_COLS = [
    "has_prior_rrp_usage",
    "coverage",
    "retailerName",
    "deviceType",
    "channel",
    "claimType",
    "country",
    "turnOnOff",
    "touchScreen",
    "smashed",
    "frontCamera",
    "backCamera",
    "frontOrBackCamera",
    "audio",
    "mic",
    "buttons",
]

EXCLUDED_COLS = ["other", "issueDesc"]


def pct(n, total):
    return f"{(100.0 * n / total):.2f}%" if total else "N/A"


def main():
    lines = []
    df = pd.read_excel(DATA_PATH)
    n_rows, n_cols = df.shape

    # Determine which requested columns are present / missing.
    present_num = [c for c in NUM_COLS if c in df.columns]
    present_cat = [c for c in CAT_COLS if c in df.columns]
    missing_num = [c for c in NUM_COLS if c not in df.columns]
    missing_cat = [c for c in CAT_COLS if c not in df.columns]
    missing_excluded = [c for c in EXCLUDED_COLS if c not in df.columns]
    target_present = TARGET in df.columns

    # --- 1. Dataset overview --------------------------------------------
    lines.append("# Claim Approval Feature Dataset — EDA Report\n")
    lines.append("## 1. Dataset Overview\n")
    lines.append(f"- **File path:** `{DATA_PATH}`")
    lines.append(f"- **Number of rows:** {n_rows}")
    lines.append(f"- **Number of columns:** {n_cols}")
    lines.append(f"- **Target column:** `{TARGET}`"
                 + ("" if target_present else "  ⚠️ **NOT FOUND in dataset**"))

    lines.append("\n### Target Value Distribution (`status`)\n")
    if target_present:
        vc = df[TARGET].value_counts(dropna=False)
        lines.append("| Value | Count | Percentage |")
        lines.append("|---|---|---|")
        for val, cnt in vc.items():
            lines.append(f"| {val} | {cnt} | {pct(cnt, n_rows)} |")
        n_missing_t = int(df[TARGET].isna().sum())
        if n_missing_t:
            lines.append(f"| *(missing)* | {n_missing_t} | {pct(n_missing_t, n_rows)} |")
    else:
        lines.append("⚠️ Target column `status` is not present in the dataset.")

    lines.append("\n### Features Included in the EDA\n")
    lines.append(f"- **Numerical features ({len(present_num)}):** "
                 + ", ".join(f"`{c}`" for c in present_num))
    lines.append(f"- **Categorical features ({len(present_cat)}):** "
                 + ", ".join(f"`{c}`" for c in present_cat))

    lines.append("\n### Columns Excluded from the EDA\n")
    lines.append("- " + ", ".join(f"`{c}`" for c in EXCLUDED_COLS)
                 + " (explicitly excluded by request)")

    # Report any missing requested columns.
    if missing_num or missing_cat or missing_excluded or not target_present:
        lines.append("\n### ⚠️ Missing / Unverified Columns\n")
        if missing_num:
            lines.append("- **Requested numerical features not found:** "
                         + ", ".join(f"`{c}`" for c in missing_num))
        if missing_cat:
            lines.append("- **Requested categorical features not found:** "
                         + ", ".join(f"`{c}`" for c in missing_cat))
        if missing_excluded:
            lines.append("- **Columns intended for exclusion not found:** "
                         + ", ".join(f"`{c}`" for c in missing_excluded))
        if not target_present:
            lines.append(f"- **Target column `{TARGET}` not found.**")
    else:
        lines.append("\n_All requested feature, excluded, and target columns "
                     "were found in the dataset._")

    # --- 2. Data type summary -------------------------------------------
    lines.append("\n## 2. Data Type Summary\n")
    lines.append("| Feature | Expected Type | Actual Pandas dtype | Type Match? |")
    lines.append("|---|---|---|---|")
    for c in present_num:
        actual = str(df[c].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(df[c])
        lines.append(f"| `{c}` | Numerical | {actual} | "
                     f"{'✅ Yes' if is_numeric else '⚠️ No (not numeric)'} |")
    for c in present_cat:
        actual = str(df[c].dtype)
        # Categorical is acceptable as object/category/bool; numeric-coded
        # categoricals (0/1 flags) are flagged as numeric-stored categoricals.
        is_numeric = pd.api.types.is_numeric_dtype(df[c])
        if pd.api.types.is_object_dtype(df[c]) or isinstance(
                df[c].dtype, pd.CategoricalDtype):
            match = "✅ Yes"
        elif is_numeric:
            match = "⚠️ Numeric-stored categorical"
        else:
            match = "✅ Yes"
        lines.append(f"| `{c}` | Categorical | {actual} | {match} |")

    # --- 3. Missing value summary ---------------------------------------
    lines.append("\n## 3. Missing Value Summary\n")
    lines.append("| Feature | Type | Missing Count | Missing % |")
    lines.append("|---|---|---|---|")
    for c in present_num:
        m = int(df[c].isna().sum())
        lines.append(f"| `{c}` | Numerical | {m} | {pct(m, n_rows)} |")
    for c in present_cat:
        m = int(df[c].isna().sum())
        lines.append(f"| `{c}` | Categorical | {m} | {pct(m, n_rows)} |")

    # --- 4. Numerical feature EDA ---------------------------------------
    lines.append("\n## 4. Numerical Feature EDA\n")
    if present_num:
        lines.append("| Feature | dtype | Non-null | Missing | Missing % | Mean | "
                     "Std | Min | 25% | Median | 75% | Max |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in present_num:
            s = pd.to_numeric(df[c], errors="coerce")
            non_null = int(s.notna().sum())
            miss = int(s.isna().sum())

            def f(x):
                return "N/A" if pd.isna(x) else f"{x:.4g}"

            lines.append(
                f"| `{c}` | {df[c].dtype} | {non_null} | {miss} | "
                f"{pct(miss, n_rows)} | {f(s.mean())} | {f(s.std())} | "
                f"{f(s.min())} | {f(s.quantile(0.25))} | {f(s.median())} | "
                f"{f(s.quantile(0.75))} | {f(s.max())} |"
            )
    else:
        lines.append("_No requested numerical features were found in the dataset._")

    # --- 5. Categorical feature EDA -------------------------------------
    lines.append("\n## 5. Categorical Feature EDA\n")
    if present_cat:
        for c in present_cat:
            s = df[c]
            non_null = int(s.notna().sum())
            miss = int(s.isna().sum())
            nunique = int(s.nunique(dropna=True))
            uniques = s.dropna().unique().tolist()

            lines.append(f"### `{c}`\n")
            lines.append(f"- **Data type:** {s.dtype}")
            lines.append(f"- **Non-null values:** {non_null}")
            lines.append(f"- **Missing values:** {miss} ({pct(miss, n_rows)})")
            lines.append(f"- **Unique values:** {nunique}")
            uniq_str = ", ".join(f"`{u}`" for u in uniques)
            lines.append(f"- **List of unique values:** {uniq_str if uniq_str else '(none)'}")
            lines.append("\n**Frequency distribution:**\n")
            lines.append("| Value | Count | Percentage |")
            lines.append("|---|---|---|")
            vc = s.value_counts(dropna=False)
            for val, cnt in vc.items():
                label = "*(missing)*" if pd.isna(val) else val
                lines.append(f"| {label} | {cnt} | {pct(cnt, n_rows)} |")
            lines.append("")
    else:
        lines.append("_No requested categorical features were found in the dataset._")

    # --- 6. Notes and observations --------------------------------------
    lines.append("\n## 6. Notes and Observations\n")
    notes = []

    if target_present:
        vc = df[TARGET].value_counts()
        if len(vc) >= 2:
            ratio = vc.iloc[0] / vc.iloc[-1]
            notes.append(
                f"**Target imbalance:** `{TARGET}` is imbalanced — "
                f"`{vc.index[0]}` ({pct(vc.iloc[0], n_rows)}) vs "
                f"`{vc.index[-1]}` ({pct(vc.iloc[-1], n_rows)}), "
                f"a ratio of ~{ratio:.1f}:1. Use stratified splits and "
                f"imbalance-aware metrics (e.g. ROC-AUC, PR-AUC, F1)."
            )

    # Missing-value-heavy features (>30%).
    heavy = []
    for c in present_num + present_cat:
        m_pct = 100.0 * df[c].isna().sum() / n_rows if n_rows else 0
        if m_pct > 30:
            heavy.append((c, m_pct))
    if heavy:
        heavy.sort(key=lambda x: -x[1])
        notes.append(
            "**Missing-value-heavy features (>30% missing):** "
            + ", ".join(f"`{c}` ({p:.1f}%)" for c, p in heavy)
            + ". Consider whether blanks are true nulls or implicit zeros "
            "(common for the device-damage flag columns)."
        )

    # High-cardinality categoricals (>20 unique).
    high_card = [(c, int(df[c].nunique(dropna=True)))
                 for c in present_cat if df[c].nunique(dropna=True) > 20]
    if high_card:
        notes.append(
            "**High-cardinality categorical features (>20 unique):** "
            + ", ".join(f"`{c}` ({n})" for c, n in high_card)
            + ". These may need grouping/encoding strategies (e.g. frequency "
            "or target encoding) rather than one-hot encoding."
        )

    # Highly imbalanced categoricals (top value >95%).
    dominant = []
    for c in present_cat:
        s = df[c].dropna()
        if len(s):
            top_share = 100.0 * s.value_counts().iloc[0] / len(s)
            if top_share > 95:
                dominant.append((c, s.value_counts().index[0], top_share))
    if dominant:
        notes.append(
            "**Highly imbalanced categorical features (single value >95%):** "
            + ", ".join(f"`{c}` ('{v}' = {p:.1f}%)" for c, v, p in dominant)
            + ". These carry little signal and may be near-constant."
        )

    # Numeric-stored categoricals.
    numeric_cats = [c for c in present_cat if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cats:
        notes.append(
            "**Numeric-stored categorical features:** "
            + ", ".join(f"`{c}`" for c in numeric_cats)
            + ". Stored as numbers but represent categories/flags; cast to "
            "category before modeling so they are not treated as continuous."
        )

    # Non-numeric numerical features.
    bad_num = [c for c in present_num if not pd.api.types.is_numeric_dtype(df[c])]
    if bad_num:
        notes.append(
            "**Unexpected dtype for numerical features:** "
            + ", ".join(f"`{c}`" for c in bad_num)
            + " are not stored as numeric and may need cleaning/conversion."
        )

    if not notes:
        notes.append("No major data-quality issues detected among the "
                     "selected features.")

    for note in notes:
        lines.append(f"- {note}")

    lines.append("")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"Report written to {OUTPUT_PATH} ({n_rows} rows, {n_cols} cols)")


if __name__ == "__main__":
    main()
