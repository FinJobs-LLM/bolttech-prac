# Claim Approval Feature Dataset — EDA Report

## 1. Dataset Overview

- **File path:** `data/preprocessed_data/claim_approval_feature_dataset.xlsx`
- **Number of rows:** 2880
- **Number of columns:** 28
- **Target column:** `status`

### Target Value Distribution (`status`)

| Value | Count | Percentage |
|---|---|---|
| Completed | 2427 | 84.27% |
| Declined | 453 | 15.73% |

### Features Included in the EDA

- **Numerical features (9):** `excessFee`, `rrp`, `oldBalanceRRP`, `remaining_rrp_ratio`, `used_rrp_amount`, `policy_duration_months`, `policy_start_year`, `policy_start_month`, `days_from_purchase_to_policy`
- **Categorical features (16):** `has_prior_rrp_usage`, `coverage`, `retailerName`, `deviceType`, `channel`, `claimType`, `country`, `turnOnOff`, `touchScreen`, `smashed`, `frontCamera`, `backCamera`, `frontOrBackCamera`, `audio`, `mic`, `buttons`

### Columns Excluded from the EDA

- `other`, `issueDesc` (explicitly excluded by request)

_All requested feature, excluded, and target columns were found in the dataset._

## 2. Data Type Summary

| Feature | Expected Type | Actual Pandas dtype | Type Match? |
|---|---|---|---|
| `excessFee` | Numerical | float64 | ✅ Yes |
| `rrp` | Numerical | float64 | ✅ Yes |
| `oldBalanceRRP` | Numerical | float64 | ✅ Yes |
| `remaining_rrp_ratio` | Numerical | float64 | ✅ Yes |
| `used_rrp_amount` | Numerical | float64 | ✅ Yes |
| `policy_duration_months` | Numerical | int64 | ✅ Yes |
| `policy_start_year` | Numerical | int64 | ✅ Yes |
| `policy_start_month` | Numerical | int64 | ✅ Yes |
| `days_from_purchase_to_policy` | Numerical | int64 | ✅ Yes |
| `has_prior_rrp_usage` | Categorical | bool | ⚠️ Numeric-stored categorical |
| `coverage` | Categorical | str | ✅ Yes |
| `retailerName` | Categorical | str | ✅ Yes |
| `deviceType` | Categorical | str | ✅ Yes |
| `channel` | Categorical | str | ✅ Yes |
| `claimType` | Categorical | str | ✅ Yes |
| `country` | Categorical | str | ✅ Yes |
| `turnOnOff` | Categorical | float64 | ⚠️ Numeric-stored categorical |
| `touchScreen` | Categorical | float64 | ⚠️ Numeric-stored categorical |
| `smashed` | Categorical | float64 | ⚠️ Numeric-stored categorical |
| `frontCamera` | Categorical | float64 | ⚠️ Numeric-stored categorical |
| `backCamera` | Categorical | float64 | ⚠️ Numeric-stored categorical |
| `frontOrBackCamera` | Categorical | float64 | ⚠️ Numeric-stored categorical |
| `audio` | Categorical | float64 | ⚠️ Numeric-stored categorical |
| `mic` | Categorical | float64 | ⚠️ Numeric-stored categorical |
| `buttons` | Categorical | float64 | ⚠️ Numeric-stored categorical |

## 3. Missing Value Summary

| Feature | Type | Missing Count | Missing % |
|---|---|---|---|
| `excessFee` | Numerical | 6 | 0.21% |
| `rrp` | Numerical | 0 | 0.00% |
| `oldBalanceRRP` | Numerical | 0 | 0.00% |
| `remaining_rrp_ratio` | Numerical | 3 | 0.10% |
| `used_rrp_amount` | Numerical | 0 | 0.00% |
| `policy_duration_months` | Numerical | 0 | 0.00% |
| `policy_start_year` | Numerical | 0 | 0.00% |
| `policy_start_month` | Numerical | 0 | 0.00% |
| `days_from_purchase_to_policy` | Numerical | 0 | 0.00% |
| `has_prior_rrp_usage` | Categorical | 0 | 0.00% |
| `coverage` | Categorical | 0 | 0.00% |
| `retailerName` | Categorical | 361 | 12.53% |
| `deviceType` | Categorical | 0 | 0.00% |
| `channel` | Categorical | 0 | 0.00% |
| `claimType` | Categorical | 0 | 0.00% |
| `country` | Categorical | 0 | 0.00% |
| `turnOnOff` | Categorical | 204 | 7.08% |
| `touchScreen` | Categorical | 203 | 7.05% |
| `smashed` | Categorical | 1877 | 65.17% |
| `frontCamera` | Categorical | 222 | 7.71% |
| `backCamera` | Categorical | 222 | 7.71% |
| `frontOrBackCamera` | Categorical | 1877 | 65.17% |
| `audio` | Categorical | 222 | 7.71% |
| `mic` | Categorical | 222 | 7.71% |
| `buttons` | Categorical | 221 | 7.67% |

## 4. Numerical Feature EDA

| Feature | dtype | Non-null | Missing | Missing % | Mean | Std | Min | 25% | Median | 75% | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `excessFee` | float64 | 2874 | 6 | 0.21% | 387.1 | 446.5 | 0 | 59 | 139 | 619 | 3629 |
| `rrp` | float64 | 2880 | 0 | 0.00% | 6926 | 7300 | 0 | 1349 | 1799 | 1.449e+04 | 4.389e+04 |
| `oldBalanceRRP` | float64 | 2880 | 0 | 0.00% | 6782 | 7214 | -4861 | 1349 | 1799 | 1.399e+04 | 4.389e+04 |
| `remaining_rrp_ratio` | float64 | 2877 | 3 | 0.10% | 0.9803 | 0.1116 | -2.587 | 1 | 1 | 1 | 1 |
| `used_rrp_amount` | float64 | 2880 | 0 | 0.00% | 144.3 | 891.3 | 0 | 0 | 0 | 0 | 1.599e+04 |
| `policy_duration_months` | int64 | 2880 | 0 | 0.00% | 13.93 | 5.344 | 6 | 12 | 12 | 12 | 24 |
| `policy_start_year` | int64 | 2880 | 0 | 0.00% | 2023 | 0.62 | 2021 | 2022 | 2023 | 2023 | 2024 |
| `policy_start_month` | int64 | 2880 | 0 | 0.00% | 5.663 | 3.272 | 1 | 3 | 6 | 8 | 12 |
| `days_from_purchase_to_policy` | int64 | 2880 | 0 | 0.00% | 12.84 | 72.61 | -260 | 0 | 0 | 0 | 366 |

## 5. Categorical Feature EDA

### `has_prior_rrp_usage`

- **Data type:** bool
- **Non-null values:** 2880
- **Missing values:** 0 (0.00%)
- **Unique values:** 2
- **List of unique values:** `False`, `True`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| False | 2698 | 93.68% |
| True | 182 | 6.32% |

### `coverage`

- **Data type:** str
- **Non-null values:** 2880
- **Missing values:** 0 (0.00%)
- **Unique values:** 2
- **List of unique values:** `ADLD/THEFT`, `ADLD`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| ADLD | 2211 | 76.77% |
| ADLD/THEFT | 669 | 23.23% |

### `retailerName`

- **Data type:** str
- **Non-null values:** 2519
- **Missing values:** 361 (12.53%)
- **Unique values:** 9
- **List of unique values:** `WUAWEI eStore`, `SWEDEN ESTORE BULK UPLOAD`, `WUAWEI SCPUserPortal`, `PSFM RETAIL ELECTRONICS`, `EVOLLIS`, `bolttech D2C`, `BREDA SES`, `UTRECHT SES`, `FINLAND ESTORE BULK UPLOAD`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| WUAWEI eStore | 1797 | 62.40% |
| WUAWEI SCPUserPortal | 380 | 13.19% |
| *(missing)* | 361 | 12.53% |
| SWEDEN ESTORE BULK UPLOAD | 204 | 7.08% |
| PSFM RETAIL ELECTRONICS | 70 | 2.43% |
| FINLAND ESTORE BULK UPLOAD | 40 | 1.39% |
| EVOLLIS | 21 | 0.73% |
| BREDA SES | 4 | 0.14% |
| bolttech D2C | 2 | 0.07% |
| UTRECHT SES | 1 | 0.03% |

### `deviceType`

- **Data type:** str
- **Non-null values:** 2880
- **Missing values:** 0 (0.00%)
- **Unique values:** 5
- **List of unique values:** `SMARTPHONES`, `TABLETS`, `WEARABLES`, `LAPTOPS`, `EARBUDS`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| SMARTPHONES | 2678 | 92.99% |
| TABLETS | 89 | 3.09% |
| WEARABLES | 62 | 2.15% |
| LAPTOPS | 50 | 1.74% |
| EARBUDS | 1 | 0.03% |

### `channel`

- **Data type:** str
- **Non-null values:** 2880
- **Missing values:** 0 (0.00%)
- **Unique values:** 5
- **List of unique values:** `Online Portal`, `Phone Call`, `Email`, `Whatsapp`, `Facebook`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| Online Portal | 2687 | 93.30% |
| Email | 133 | 4.62% |
| Phone Call | 57 | 1.98% |
| Whatsapp | 2 | 0.07% |
| Facebook | 1 | 0.03% |

### `claimType`

- **Data type:** str
- **Non-null values:** 2880
- **Missing values:** 0 (0.00%)
- **Unique values:** 3
- **List of unique values:** `Theft`, `Accidental Damage`, `Liquid Damage`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| Accidental Damage | 2715 | 94.27% |
| Theft | 98 | 3.40% |
| Liquid Damage | 67 | 2.33% |

### `country`

- **Data type:** str
- **Non-null values:** 2880
- **Missing values:** 0 (0.00%)
- **Unique values:** 3
- **List of unique values:** `SE`, `NL`, `FI`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| NL | 1533 | 53.23% |
| SE | 1211 | 42.05% |
| FI | 136 | 4.72% |

### `turnOnOff`

- **Data type:** float64
- **Non-null values:** 2676
- **Missing values:** 204 (7.08%)
- **Unique values:** 2
- **List of unique values:** `1.0`, `0.0`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| 1.0 | 2361 | 81.98% |
| 0.0 | 315 | 10.94% |
| *(missing)* | 204 | 7.08% |

### `touchScreen`

- **Data type:** float64
- **Non-null values:** 2677
- **Missing values:** 203 (7.05%)
- **Unique values:** 2
- **List of unique values:** `1.0`, `0.0`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| 0.0 | 1490 | 51.74% |
| 1.0 | 1187 | 41.22% |
| *(missing)* | 203 | 7.05% |

### `smashed`

- **Data type:** float64
- **Non-null values:** 1003
- **Missing values:** 1877 (65.17%)
- **Unique values:** 1
- **List of unique values:** `0.0`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| *(missing)* | 1877 | 65.17% |
| 0.0 | 1003 | 34.83% |

### `frontCamera`

- **Data type:** float64
- **Non-null values:** 2658
- **Missing values:** 222 (7.71%)
- **Unique values:** 2
- **List of unique values:** `1.0`, `0.0`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| 1.0 | 2234 | 77.57% |
| 0.0 | 424 | 14.72% |
| *(missing)* | 222 | 7.71% |

### `backCamera`

- **Data type:** float64
- **Non-null values:** 2658
- **Missing values:** 222 (7.71%)
- **Unique values:** 2
- **List of unique values:** `1.0`, `0.0`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| 1.0 | 2198 | 76.32% |
| 0.0 | 460 | 15.97% |
| *(missing)* | 222 | 7.71% |

### `frontOrBackCamera`

- **Data type:** float64
- **Non-null values:** 1003
- **Missing values:** 1877 (65.17%)
- **Unique values:** 1
- **List of unique values:** `0.0`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| *(missing)* | 1877 | 65.17% |
| 0.0 | 1003 | 34.83% |

### `audio`

- **Data type:** float64
- **Non-null values:** 2658
- **Missing values:** 222 (7.71%)
- **Unique values:** 2
- **List of unique values:** `1.0`, `0.0`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| 1.0 | 2293 | 79.62% |
| 0.0 | 365 | 12.67% |
| *(missing)* | 222 | 7.71% |

### `mic`

- **Data type:** float64
- **Non-null values:** 2658
- **Missing values:** 222 (7.71%)
- **Unique values:** 2
- **List of unique values:** `1.0`, `0.0`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| 1.0 | 2276 | 79.03% |
| 0.0 | 382 | 13.26% |
| *(missing)* | 222 | 7.71% |

### `buttons`

- **Data type:** float64
- **Non-null values:** 2659
- **Missing values:** 221 (7.67%)
- **Unique values:** 2
- **List of unique values:** `0.0`, `1.0`

**Frequency distribution:**

| Value | Count | Percentage |
|---|---|---|
| 0.0 | 2652 | 92.08% |
| *(missing)* | 221 | 7.67% |
| 1.0 | 7 | 0.24% |


## 6. Notes and Observations

- **Target imbalance:** `status` is imbalanced — `Completed` (84.27%) vs `Declined` (15.73%), a ratio of ~5.4:1. Use stratified splits and imbalance-aware metrics (e.g. ROC-AUC, PR-AUC, F1).
- **Missing-value-heavy features (>30% missing):** `smashed` (65.2%), `frontOrBackCamera` (65.2%). Consider whether blanks are true nulls or implicit zeros (common for the device-damage flag columns).
- **Highly imbalanced categorical features (single value >95%):** `smashed` ('0.0' = 100.0%), `frontOrBackCamera` ('0.0' = 100.0%), `buttons` ('0.0' = 99.7%). These carry little signal and may be near-constant.
- **Numeric-stored categorical features:** `has_prior_rrp_usage`, `turnOnOff`, `touchScreen`, `smashed`, `frontCamera`, `backCamera`, `frontOrBackCamera`, `audio`, `mic`, `buttons`. Stored as numbers but represent categories/flags; cast to category before modeling so they are not treated as continuous.
