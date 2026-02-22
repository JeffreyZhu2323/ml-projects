# Customer Churn Prediction + Retention Targeting (Telco)

End-to-end churn modeling project using **scikit-learn Logistic Regression** and **XGBoost** with:
- train / validation / test evaluation
- model comparison (ROC-AUC + PR-AUC)
- threshold policy analysis for different retention team capacities
- business-oriented interpretation (including an interpretable logistic regression benchmark)

---

## Project Goal

Predict which customers are likely to churn so a retention team can prioritize outreach more effectively.

This project is framed as a **ranking + targeting** problem, not just a binary classification problem:
- The model produces churn probabilities
- Different thresholds support different outreach capacities (e.g., top 10% highest-risk customers vs. a more balanced F1 threshold)

---

## TL;DR Results (Test Set)

I trained and compared:
- **Logistic Regression (baseline)**
- **Logistic Regression (tuned)**
- **XGBoost (tuned)**

### Model Comparison (Test Set)

| Model | ROC-AUC | PR-AUC | Summary |
|---|---:|---:|---|
| Logistic Regression (baseline) | **0.8498** | **0.6461** | Strong interpretable baseline |
| Logistic Regression (tuned) | 0.8494 | 0.6453 | Similar/slightly worse than baseline |
| **XGBoost (tuned)** | **0.8562** | **0.6744** | **Best overall performance** |

### Final Model Selection
**Selected final model: `XGBoost (tuned)`**

Why:
- Best **PR-AUC** (important for imbalanced churn prediction)
- Best **ROC-AUC**
- Better ranking quality for prioritizing limited retention outreach

### Improvement vs Logistic Baseline
- **ROC-AUC:** +0.0064
- **PR-AUC:** +0.0282

> PR-AUC is emphasized because churn is an imbalanced classification problem, and PR-AUC better reflects how well the model identifies actual churners compared with accuracy alone.

---

## Practical Operating Points (XGBoost Tuned, Test Set)

The final model outputs probabilities. Different thresholds correspond to different outreach strategies.

| Policy | Threshold | Target Rate | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Default threshold | 0.5000 | 21.58% | 0.6711 | 0.5455 | 0.6018 |
| Top 10% highest-risk customers | 0.6648 | 10.01% | **0.7730** | 0.2914 | 0.4233 |
| Best F1 threshold (selected on validation set) | 0.3910 | 30.94% | 0.5894 | **0.6872** | **0.6346** |

### Confusion Matrices (XGBoost Tuned, Test Set)

- **Threshold = 0.50**
  - `[[935, 100], [170, 204]]`

- **Top 10% threshold**
  - `[[1003, 32], [265, 109]]`

- **Best F1 threshold**
  - `[[856, 179], [117, 257]]`

### How to use these thresholds (business interpretation)
- **Top 10% threshold**: best when retention capacity is limited and you only want to contact a small, high-risk group (higher precision).
- **Best F1 threshold**: better if the team can contact more customers and wants a more balanced precision/recall tradeoff.
- **0.50 threshold**: useful default baseline, but not always the best business operating point.

---

## Modeling Approach

### 1) Logistic Regression (Baseline)
Used as an interpretable benchmark:
- clean baseline for tabular binary classification
- easy to inspect coefficient directions and magnitudes
- strong first benchmark for churn

### 2) Logistic Regression (Tuned)
Hyperparameter tuning via cross-validation (e.g., regularization strength / solver combinations).  
Result: similar performance to baseline, slightly lower ranking metrics on the test set.

### 3) XGBoost (Tuned) — Final Model
Used gradient-boosted trees to capture:
- nonlinear effects
- feature interactions
- more flexible decision boundaries than logistic regression

XGBoost achieved the best performance on both:
- **ROC-AUC**
- **PR-AUC**

---

## Evaluation Philosophy

This project evaluates model quality in two layers:

### A) Ranking Quality (model selection)
Primary metrics:
- **PR-AUC (Average Precision)**
- **ROC-AUC**

### B) Decision Policy Quality (deployment / operations)
Threshold-based metrics:
- Precision
- Recall
- F1
- Target rate
- Confusion matrix

This separation matters because:
- **model tuning** improves the probability ranking
- **threshold tuning** defines the actual business action policy

---

## Interpretability Benchmark (Logistic Regression)

Even though **XGBoost** was selected as the final predictive model, logistic regression remains valuable as an interpretable benchmark for directional insights into churn drivers.

Examples of how this is useful:
- sanity-checking whether learned relationships match intuition
- communicating model behavior to non-technical stakeholders
- comparing transparent linear effects vs. nonlinear boosted-tree performance

If included in this repo, see:
- `results/variable_affect_churn.csv` (logistic coefficient summary / feature effects)

---

## Data & Preprocessing

This project uses the **Telco Customer Churn** dataset and applies a consistent preprocessing pipeline for all models.

- **Target:** `Churn Value` (binary)
- **Features:** customer demographics, services, contract/billing info, and charges (including `Total Charges`)
- **Cleaning:** converts `Total Charges` / `Monthly Charges` to numeric; fills missing `Total Charges` with `0` for customers with `Tenure Months == 0`
- **Encoding:** maps binary fields to `0/1` and one-hot encodes multi-class categorical columns (`Internet Service`, `Contract`, `Payment Method`)
- **Splitting:** stratified train / validation / test split (same random seed for reproducibility)
- **Scaling:** standardizes continuous features (`Tenure Months`, `Monthly Charges`, `Total Charges`) using **training-set statistics only** (then applies to val/test to avoid leakage)

The data loader returns:
`X_train, y_train, X_val, y_val, X_test, y_test`

---

## Saved Outputs

This project stores evaluation outputs in the `results/` directory, including:

- `results/performance_metrics`  
  JSON metrics for:
  - logistic regression baseline
  - logistic regression tuned
  - xgboost tuned
  - threshold-based evaluations

Optional/related outputs (if generated in your local version):
- grid search CV result tables (`*.csv`)
- plots (ROC / PR curves)
- feature importance summaries

> Note: In my current setup, the metrics file is saved as `performance_metrics` (JSON content, no extension). You can rename it to `performance_metrics.json` if you prefer.

---

## How to Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
