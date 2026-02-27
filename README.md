# ML Projects 

[![CI](https://github.com/JeffreyZhu2323/ml-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/JeffreyZhu2323/ml-projects/actions/workflows/ci.yml)

Curated ML mini-projects built in PyTorch, focused on clean training/evaluation pipelines and reproducible structure. Each project has its own directory with its own README; code lives in `src/` (run scripts from inside `src/`), and outputs go to `results/` and `reports/`.
**Python:** 3.10.0

## Projects
- **[California Housing — Linear Regression (PyTorch)](./california_housing_linear_regression/)** — from-scratch training loop (w, b, manual GD) + feature engineering (quantile-plot EDA, cutoffs/hinges) + ridge sweep; test R² 0.59 (≈13% gain over raw features)
- **[Customer Churn — Logistic Regression + XGBoost](./customer_churn_prediction/)** — Telco churn (LogReg vs XGBoost, CV tuning), PR-AUC/calibration, retention-threshold policies; XGBoost PR-AUC 0.67
  
## Setup

### Create a virtual environment (optional but recommended)

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
