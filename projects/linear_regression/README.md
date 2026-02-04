# California Housing Linear Regression (PyTorch)

Linear regression baseline on the California Housing dataset implemented in **PyTorch**, focusing on clean, reproducible ML engineering practices (train/val/test split, train-only standardization, metric tracking, checkpointing, and ridge regularization sweep).

## What this project demonstrates
- **From-scratch linear regression in PyTorch** (explicit `w, b` parameters)
- **Train/Val/Test workflow** with reproducible splits (fixed split & pytorch training seeds)
- Metrics: **MSE, RMSE, R²**
- **Best validation checkpointing**
- **Ridge (L2) regularization sweep** to select λ by validation performance
- Simple plotting of learning curves

## Dataset
- **Source:** `sklearn.datasets.fetch_california_housing`
- **Target:** `MedHouseVal`
- **Features used (6):**
  - `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`

## Project structure
- `config.py`  
  Hyperparameters + metric functions (`mse_loss`, `rmse_loss`, `r2`) + paths.
- `data_loading.py`  
  Loads dataset, selects features, creates train/val/test splits, standardizes using train stats.
- `training.py`  
  Trains linear regression with gradient descent, logs metrics, saves best-val checkpoint.
- `ridge_regularization_sweep.py`  
  Sweeps λ over a list of values and records best validation MSE for each λ (ridge).
- `evaluation.py`  
  Loads saved checkpoint and evaluates on the test set.
- `loss_graph.py`  
  Reads `loss_graph.csv` and plots curves.

## Results 
- Validation MSE around **~0.645**
- Validation R² around **~0.54**
- Ridge sweep may give a small improvement (e.g., best near **λ = 0.01**)

## How to run

### 1) Install dependencies
From repo root:
```bash
pip install -r requirements.txt
