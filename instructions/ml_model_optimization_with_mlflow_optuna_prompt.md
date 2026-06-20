You are a senior machine learning engineer and full-stack AI application developer.

I want to build the best-performing machine learning model to predict `status` with two classes: `Completed` and `Declined`.

The dataset is located at:

`/home/ubuntu/bolttech-prac/data/preprocessed_data/claim_approval_feature_dataset_v2.xlsx`

The target column is:

`status`

The positive class should be:

`Declined = 1`

The negative class should be:

`Completed = 0`

Columns `other` and `issueDesc` should be excluded from the feature set if they exist.

The goal is to train, optimize, compare, and select the best-performing model among the following models:

1. Random Forest
2. XGBoost
3. LightGBM
4. CatBoost

The dataset has an imbalanced target distribution. The approximate class ratio is:

* Completed: majority class
* Declined: minority class
* Recommended imbalance weight: approximately `5.36`

Please implement the full machine learning workflow with the following requirements.

---

## 1. Data Loading and Validation

Load the Excel file from:

`/home/ubuntu/bolttech-prac/data/preprocessed_data/claim_approval_feature_dataset_v2.xlsx`

Check and report:

* Dataset shape
* Column names
* Target class distribution
* Missing values by column
* Numerical and categorical feature columns
* Any columns excluded from training

Use `status` as the target column.

Exclude:

```python
["other", "issueDesc"]
```

if they exist.

---

## 2. Target Encoding

Encode the target as follows:

```python
Completed -> 0
Declined -> 1
```

Make sure all metrics treat `Declined` as the positive class.

---

## 3. Train / Validation / Test Split

Use a stratified split so that the class ratio of `Completed` and `Declined` is preserved across all datasets.

Create:

* Training set
* Validation set
* Test set

For example:

```python
train: 70%
validation: 15%
test: 15%
```

Use `stratify=y`.

The validation set should be used for model selection, hyperparameter optimization, threshold tuning, and early stopping where applicable.

The test set should only be used for the final unbiased evaluation of the selected best model.

---

## 4. Preprocessing

Build preprocessing pipelines that correctly handle both numerical and categorical features.

For numerical features:

* Impute missing values using median
* Scaling is optional for tree-based models, but include it only if necessary

For categorical features:

* Impute missing values with `"Unknown"`
* Convert categorical features to string
* For Random Forest, XGBoost, and LightGBM, use appropriate encoding such as One-Hot Encoding or Ordinal Encoding
* For CatBoost, use CatBoost’s native categorical feature handling when possible

Make sure preprocessing is fitted only on the training set to avoid data leakage.

---

## 5. Baseline Models

Train baseline versions of all four models before hyperparameter optimization.

The baseline models should include:

### Random Forest

Use:

```python
class_weight="balanced"
```

### XGBoost

Use:

```python
scale_pos_weight=5.36
```

### LightGBM

Use:

```python
scale_pos_weight=5.36
```

Do not use `is_unbalance=True` at the same time as `scale_pos_weight`.

### CatBoost

Compare one of the following imbalance strategies:

```python
class_weights={"Completed": 1, "Declined": 5.36}
```

or:

```python
auto_class_weights="Balanced"
```

Make sure CatBoost correctly recognizes categorical features.

---

## 6. Hyperparameter Optimization with Optuna

Use Optuna to optimize hyperparameters for each model.

The optimization objective should prioritize imbalanced binary classification performance.

Do not optimize for Accuracy as the primary metric.

Use one of the following as the primary optimization metric:

* PR-AUC
* F1 score for Declined
* A combined score using PR-AUC, Recall for Declined, Precision for Declined, and Balanced Accuracy

Recommended primary objective:

```text
Maximize PR-AUC on the validation set
```

Also track:

* F1 for Declined
* Recall for Declined
* Precision for Declined
* Balanced Accuracy
* ROC-AUC
* Accuracy
* Confusion Matrix
* False Positives
* False Negatives

For each model, tune meaningful hyperparameters.

Examples:

### Random Forest

Tune:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`
* `class_weight`

### XGBoost

Tune:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`
* `colsample_bytree`
* `min_child_weight`
* `gamma`
* `reg_alpha`
* `reg_lambda`
* `scale_pos_weight`

### LightGBM

Tune:

* `n_estimators`
* `num_leaves`
* `max_depth`
* `learning_rate`
* `subsample`
* `colsample_bytree`
* `min_child_samples`
* `reg_alpha`
* `reg_lambda`
* `scale_pos_weight`

### CatBoost

Tune:

* `iterations`
* `depth`
* `learning_rate`
* `l2_leaf_reg`
* `bagging_temperature`
* `random_strength`
* `border_count`
* `class_weights` or `auto_class_weights`

Use a reproducible random seed.

---

## 7. MLflow Experiment Tracking

Use MLflow to track all experiments.

For every run, log:

* Model name
* Baseline or optimized model type
* Hyperparameters
* Imbalance strategy
* Preprocessing strategy
* Class distribution
* Validation metrics
* Test metrics
* PR-AUC
* ROC-AUC
* F1 for Declined
* Recall for Declined
* Precision for Declined
* Balanced Accuracy
* Accuracy
* Confusion matrix image
* Precision-Recall curve image
* ROC curve image
* Feature importance image where available
* Best threshold selected from validation set
* Final trained model artifact

Use MLflow nested runs if appropriate:

* Parent run: model family, such as XGBoost
* Child runs: individual Optuna trials

At the end, register or save the best model based on validation PR-AUC and final test performance.

The MLflow UI should allow the user to understand:

* Which model performed best
* Which hyperparameters were selected
* Why the selected model is better than the alternatives
* How the imbalance strategy affected performance
* Whether the model is good at detecting the minority class `Declined`

---

## 8. Threshold Tuning

Do not rely only on the default classification threshold of `0.5`.

For each optimized model, tune the decision threshold on the validation set.

Search thresholds from 0.05 to 0.95.

For each threshold, calculate:

* Precision for Declined
* Recall for Declined
* F1 for Declined
* Balanced Accuracy
* Confusion Matrix

Choose the threshold that best fits the business goal.

By default, choose the threshold that maximizes F1 for Declined.

Also show how different thresholds affect precision and recall.

The final test evaluation should use the selected threshold from the validation set.

---

## 9. Model Comparison Report

Create a final comparison table with all baseline and optimized models.

The table should include:

* Model name
* Baseline or optimized
* Imbalance strategy
* Best hyperparameters
* Best threshold
* Validation PR-AUC
* Test PR-AUC
* Test ROC-AUC
* Test F1 for Declined
* Test Recall for Declined
* Test Precision for Declined
* Test Balanced Accuracy
* Test Accuracy
* False Positives
* False Negatives

Explain the final model selection in plain English.

The explanation should answer:

* Why this model was selected
* Why Accuracy was not the main metric
* How PR-AUC is useful for imbalanced classification
* Whether the model is good at detecting `Declined`
* What trade-off exists between Precision and Recall
* How the selected threshold affects the final business decision
* Which hyperparameters were important
* How MLflow and Optuna help make the model selection transparent and explainable

---

## 10. Front-End Dashboard Design

Design and implement a front-end dashboard using React or JavaScript.

The dashboard should visually explain the model optimization and selection process to non-technical or semi-technical users.

The front-end should include the following pages or sections:

### A. Overview Page

Show:

* Dataset name
* Number of rows
* Number of features
* Target class distribution
* Completed vs Declined imbalance ratio
* Train / validation / test split ratio

Use visual components such as:

* Class distribution bar chart
* Pie chart or donut chart
* Summary cards

### B. Model Leaderboard

Show a sortable comparison table of all trained models.

Columns should include:

* Model
* Baseline / Optimized
* Imbalance strategy
* PR-AUC
* F1 for Declined
* Recall for Declined
* Precision for Declined
* Balanced Accuracy
* Accuracy
* Best threshold

Highlight the best model.

Accuracy should be shown, but visually de-emphasized compared to PR-AUC, F1, Recall, Precision, and Balanced Accuracy.

### C. MLflow / Optuna Tracking Page

Show:

* Hyperparameter search history
* Best trial
* Best hyperparameters
* Metric changes across trials
* Parallel coordinate plot if possible
* Parameter importance plot if possible

The user should be able to understand why the selected hyperparameters were chosen.

### D. Metric Explanation Page

Explain the meaning of:

* PR-AUC
* Precision for Declined
* Recall for Declined
* F1 for Declined
* Balanced Accuracy
* Confusion Matrix
* ROC-AUC
* Accuracy

Use simple business-friendly explanations.

For example:

* Recall for Declined: “Among truly declined claims, how many did the model correctly detect?”
* Precision for Declined: “Among claims predicted as declined, how many were actually declined?”
* PR-AUC: “How well the model separates the minority class across different thresholds.”

### E. Threshold Tuning Page

Show:

* Precision-Recall curve
* Threshold vs Precision
* Threshold vs Recall
* Threshold vs F1
* Confusion matrix at selected threshold
* Recommended threshold

Allow the user to visually understand the trade-off between catching more Declined cases and avoiding false alarms.

### F. Final Model Explanation Page

Show:

* Selected best model
* Selected hyperparameters
* Selected threshold
* Final test performance
* Feature importance
* Confusion matrix
* Plain-English explanation of why this model was selected

### G. Prediction Demo Page

Allow the user to input feature values and get:

* Predicted class
* Probability of Completed
* Probability of Declined
* Decision threshold used
* Explanation of the prediction if possible

---

## 11. Back-End API

Create a simple API using FastAPI.

The API should provide:

### `/predict`

Input:

* Feature values in JSON format

Output:

* Predicted class
* Probability of Completed
* Probability of Declined
* Threshold used

### `/model-summary`

Output:

* Best model name
* Best hyperparameters
* Test metrics
* Threshold
* Imbalance strategy

### `/model-comparison`

Output:

* Comparison table of all trained models

### `/threshold-analysis`

Output:

* Threshold-level precision, recall, F1, and balanced accuracy

### `/feature-importance`

Output:

* Feature importance values for the selected model

---

## 12. Project Structure

Create a clean project structure like this:

```text
bolttech-prac/
├── data/
│   └── preprocessed_data/
│       └── claim_approval_feature_dataset_v2.xlsx
├── mlruns/
├── models/
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   ├── threshold_analysis.csv
│   └── final_model_report.md
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── train_baselines.py
│   ├── optimize_optuna.py
│   ├── evaluate.py
│   ├── threshold_tuning.py
│   ├── mlflow_tracking.py
│   ├── explainability.py
│   └── serve.py
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── pages/
│   │   ├── components/
│   │   └── api/
│   └── vite.config.js
├── notebooks/
│   └── model_experiment_summary.ipynb
├── requirements.txt
└── README.md
```

---

## 13. Deliverables

Please produce:

1. Python training pipeline
2. Baseline model training code
3. Optuna optimization code
4. MLflow tracking integration
5. Threshold tuning code
6. Final model comparison report
7. Saved best model artifact
8. FastAPI serving API
9. React or JavaScript front-end dashboard
10. README with clear instructions

The README should explain:

* How to install dependencies
* How to run MLflow UI
* How to train models
* How to run Optuna optimization
* How to view model comparison results
* How to start the FastAPI server
* How to start the React front-end
* How to interpret the metrics and dashboard

---

## 14. Important Modeling Principles

Follow these principles strictly:

* Do not use the test set during hyperparameter tuning
* Do not perform preprocessing before train/validation/test split in a way that causes data leakage
* Do not oversample the validation or test set
* Use stratified splitting
* Treat `Declined` as the positive class
* Do not optimize primarily for Accuracy
* Use PR-AUC, Recall, Precision, F1, and Balanced Accuracy as the main metrics
* Track all important experiments using MLflow
* Use Optuna to make hyperparameter optimization systematic and reproducible
* Make the model selection process explainable to the user
* Make the front-end dashboard visually clear and business-friendly

---

## 15. Final Output

At the end, provide a clear final recommendation:

```text
Best Model: <model name>
Best Imbalance Strategy: <strategy>
Best Hyperparameters: <hyperparameters>
Best Threshold: <threshold>
Main Reason for Selection: <plain English explanation>
Key Risk or Limitation: <plain English explanation>
```

Also explain why this model should be trusted more than the alternatives based on validation and test metrics.
