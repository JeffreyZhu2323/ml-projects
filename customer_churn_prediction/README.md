# Customer Churn Prediction + Retention Targeting (Telco)

**End-to-end ML pipeline for predicting telco customer churn and turning model scores into actionable retention strategies.** Built with Python, scikit-learn, XGBoost, and Google Cloud BigQuery.

This project demonstrates a production-style workflow: **data ingestion (BigQuery) → preprocessing → stratified splits → model training & tuning → evaluation (ROC-AUC, PR-AUC, calibration) → threshold policies for business use.**

---

## Highlights 

- **Cloud data pipeline:** Primary data source is **Google Cloud BigQuery**; dataset is queried via SQL in GCP and loaded programmatically, with a **pandas-based backup loader** for local CSV/Excel.
- **Rigorous evaluation:** Compares Logistic Regression (interpretable baseline) vs XGBoost with **ROC-AUC**, **PR-AUC**, and **Brier score** (probability calibration); emphasizes PR-AUC for imbalanced churn.
- **Actionable output:** Threshold-based policies (e.g. “top 10% highest-risk”) with precision/recall/F1 and confusion matrices for retention outreach decisions.
- **Reproducibility:** Stratified train/validation/test splits, training-set-only scaling, and saved configs/artifacts.

---

## Results

### Model Comparison

| Model | ROC-AUC | PR-AUC | Brier score | Top 10% precision | Summary |
|---|---:|---:|---:|---:|---|
| Logistic Regression (baseline) | **0.8498** | **0.6461** | – | 0.752 | Strong interpretable baseline |
| Logistic Regression (tuned) | 0.8494 | 0.6453 | 0.1354 | 0.752 | Similar to baseline |
| **XGBoost (tuned)** | **0.8548** | **0.6773** | **0.1326** | **0.773** | **Best overall performance** |

*Metrics from test set; data loaded from BigQuery (see Data source below).*

### Final Model Selection
**Selected model: XGBoost (tuned)**

- Best **PR-AUC** (most informative for imbalanced churn prediction)
- Best **ROC-AUC**
- Best probability calibration (lowest Brier score) and better calibration curve

### Improvement vs Logistic Baseline
- **ROC-AUC:** +0.0050  
- **PR-AUC:** +0.0312  
- **Calibration:** Lower Brier score and better calibration curve for XGBoost

> PR-AUC is emphasized because churn is an imbalanced classification problem, so it is more informative than accuracy for identifying actual churners.

---

## Test Curves (Final Model)

<p align="center">
  <a href="reports/figures/xgboost_pr_curve.png">
    <img src="reports/figures/xgboost_pr_curve.png" alt="Precision-Recall Curve" width="48%"/>
  </a>
  <a href="reports/figures/xgboost_roc_curve.png">
    <img src="reports/figures/xgboost_roc_curve.png" alt="ROC Curve" width="48%"/>
  </a>
</p>

<p align="center">
  <a href="reports/figures/xgboost_calibration_curve.png">
    <img src="reports/figures/xgboost_calibration_curve.png" alt="XGBoost Calibration Curve (Reliability Diagram)" width="48%"/>
  </a>
</p>

---

## Practical Operating Points (XGBoost Tuned, Test Set)

The final model outputs churn probabilities. Different outreach thresholds trade off **coverage** (how many customers you contact) vs **precision/recall** (how many actually churn).

| Policy | Threshold | Target Rate | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Default threshold | 0.5000 | 22.71% | 0.6313 | 0.5401 | 0.5821 |
| Top 5% highest-risk customers | 0.7541 | 5.04% | **0.8592** | 0.1631 | 0.2742 |
| Top 10% highest-risk customers | 0.6841 | 10.01% | **0.7730** | 0.2914 | 0.4233 |
| Top 20% highest-risk customers | 0.5363 | 20.01% | 0.6525 | 0.4920 | 0.5610 |
| Top 30% highest-risk customers | 0.3990 | 30.02% | 0.5957 | 0.6738 | **0.6324** |
| Best F1 threshold (validation-set tuned) | 0.3615 | 32.43% | 0.5799 | **0.7086** | **0.6378** |

### Confusion Matrices (XGBoost Tuned, Test Set)
- **Threshold = 0.50** → `[[917, 118], [172, 202]]`
- **Top 10% threshold** → `[[1003, 32], [265, 109]]`
- **Best F1 threshold** → `[[843, 192], [109, 265]]`

### Business Interpretation
- **Top 5–10% thresholds:** Very high precision, lower recall — use when outreach capacity is limited and you want to contact only the highest-risk customers.
- **Top 20–30% thresholds:** More balanced precision/recall — use when you can contact more customers and want to catch a larger share of potential churners.
- **Best F1 threshold:** Tuned on validation data; similar target rate to a 30% policy but optimized for F1.
- **0.50 threshold:** Familiar default, but not necessarily optimal for business objectives.

---

## Data Source & Target

- **Primary data source:** **Google Cloud BigQuery.** The pipeline loads the Telco churn dataset from a BigQuery table (e.g. `churn_features`) via SQL. Feature preparation can be done in SQL in GCP, then the preprocessed or raw table is queried and loaded in Python.
- **Backup data source:** Local **CSV or Excel** (e.g. `Telco_customer_churn.csv` / `Telco_customer_churn.xlsx` in `data/`) using a separate loader for environments without GCP access.
- **Target:** Binary churn indicator (`Churn Value` or `churn`, 0/1).

The pipeline produces a consistent set of encoded model features after preprocessing (column names may be snake_case when using the BigQuery-derived table).

---

## Data Loading Architecture

| Module | Purpose |
|--------|--------|
| **`main_data_loading.py`** | Loads data from **Google Cloud BigQuery**: connects via `google-cloud-bigquery`, runs a SQL query (e.g. `SELECT * FROM project.dataset.table ORDER BY CustomerID`), and returns train/val/test arrays with the same split and scaling logic as below. |
| **`backup_data_loading.py`** | Loads from **local CSV or Excel** with pandas: full cleaning (e.g. `Total Charges`/`Monthly Charges` handling, missing values), binary and one-hot encoding, then same split and scaling. Use when BigQuery is not available. |

All scripts call **`load_data()`** from **`data_loading`**. The backend is chosen in **`config.py`** via **`DATA_SOURCE`**: `"bigquery"` (SQL) or `"pandas"` (local CSV/Excel). No need to change imports when switching (see [How to Run](#how-to-run)).

---

## Data & Preprocessing

The pipeline uses a consistent preprocessing approach for all models:

- **Target:** Binary churn (0/1).
- **Features:** Demographics, services, contract/billing, and charges (e.g. tenure, monthly/total charges).
- **Cleaning (backup loader):** Converts `Monthly Charges` / `Total Charges` to numeric; fills missing `Total Charges` with `0` when tenure is 0.
- **Encoding:** Binary fields mapped to 0/1; multi-class columns (e.g. Internet Service, Contract, Payment Method) one-hot encoded.
- **Splitting:** Stratified train / validation / test split with a fixed random seed.
- **Scaling:** Continuous features (e.g. tenure, monthly charges, total charges) standardized using **training-set statistics only** (applied to val/test to avoid leakage).

The data loader returns:  
`X_train, y_train, X_val, y_val, X_test, y_test`

---

## Models

### Logistic Regression (Baseline + Tuned)
- Interpretable benchmark; fast and stable for tabular binary classification.
- Coefficient-based churn driver analysis (see `results/variable_affect_churn.csv`).

### XGBoost (Tuned) — Final Model
- Captures nonlinear effects, feature interactions, and more flexible decision boundaries.
- Selected for best test ROC-AUC and PR-AUC.

---

## Interpretability Benchmark (Logistic Regression)

Logistic regression is retained as an interpretable benchmark for directional insights:

- **Output:** `results/variable_affect_churn.csv` (coefficient summary).
- **Takeaways:** Lower tenure and shorter contracts associate with higher churn risk; certain service/payment patterns (e.g. fiber optic, electronic check) with higher risk; support/security add-ons with lower churn.

---

## Saved Outputs

| Location | Contents |
|----------|----------|
| `results/performance_metrics.json` | ROC-AUC, PR-AUC, Brier score, threshold-based metrics, confusion matrices, calibration summary for all models. |
| `results/best_*_params.json` | Best hyperparameters from grid search (e.g. logistic regression, XGBoost). |
| `results/variable_affect_churn.csv` | Logistic regression coefficients / feature impact. |
| `reports/figures/` | ROC, PR, and calibration curves (e.g. XGBoost). |

---

## Project Structure

```text
customer_churn_prediction/
├── data/
│   ├── .gitkeep
│   └── Telco_customer_churn.xlsx   # optional; for backup loader
├── reports/
│   └── figures/
│       ├── xgboost_pr_curve.png
│       ├── xgboost_roc_curve.png
│       ├── xgboost_calibration_curve.png
│       └── logistic_calibration_curve.png
├── results/
│   ├── performance_metrics.json
│   ├── best_logisticreg_params.json
│   ├── best_xgboost_params.json
│   └── variable_affect_churn.csv
├── sql/                             # optional; BigQuery / exploration queries
│   ├── 01_label_distribution.sql
│   ├── 02_churn_by_segment.sql
│   └── 03_missingness.sql
├── src/
│   ├── config.py
│   ├── data_loading.py              # single load_data() entry point; backend set by config.DATA_SOURCE
│   ├── main_data_loading.py        # BigQuery loader (primary)
│   ├── backup_data_loading.py      # pandas CSV/Excel loader (backup)
│   ├── logisticreg_hyperparmtuning.py
│   ├── logisticreg_training_eval.py
│   ├── xgboost_hyperparamtuning.py
│   └── xgboost_training_eval.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Required: `scikit-learn`, `xgboost`, `pandas`, `google-cloud-bigquery` (and optionally `openpyxl` for Excel in the backup loader).

### 2. Primary path: BigQuery

1. **Google Cloud:** Create a project and enable the BigQuery API. Upload or build your Telco churn table (e.g. `churn_features` in dataset `telcocustomerchurn`).
2. **Credentials:** Set up Application Default Credentials (e.g. `gcloud auth application-default login`) or a service account key and point `GOOGLE_APPLICATION_CREDENTIALS` to it.
3. **Config:** In `src/main_data_loading.py`, set `project`, `dataset`, and `table` in `load_from_bigquery(...)` to match your GCP resource (e.g. `customerchurn-488906`, `telcocustomerchurn`, `churn_features`).
4. In `src/config.py`, set **`DATA_SOURCE = "bigquery"`** (default). Run training/eval scripts from the project root (with `src` on the path). They all call **`load_data()`** from **`data_loading`**.

### 3. Backup path: Local CSV/Excel

Place `Telco_customer_churn.csv` or `Telco_customer_churn.xlsx` in `data/`. In **`src/config.py`** set **`DATA_SOURCE = "pandas"`**. Run the same pipeline; **`data_loading.load_data()`** will use **`backup_data_loading`** and the local file with the full pandas-based cleaning and encoding.

### 4. Run the pipeline

From the project root (with `src` on `PYTHONPATH` or from within `src`):

```bash
# Hyperparameter tuning (optional)
python -m src.logisticreg_hyperparmtuning
python -m src.xgboost_hyperparamtuning

# Train and evaluate
python -m src.logisticreg_training_eval
python -m src.xgboost_training_eval
```

Artifacts (metrics, plots, saved params) will be written to `results/` and `reports/figures/`.

---

## Tech Stack

Python · scikit-learn · XGBoost · pandas · Google Cloud BigQuery · stratified evaluation · probability calibration (Brier score) · precision–recall & ROC analysis
