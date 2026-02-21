# California Housing Linear Regression (PyTorch)

Linear regression baseline on the California Housing dataset implemented in **PyTorch**, focusing on clean, reproducible ML engineering practices (train/val/test split, train-only standardization, metric tracking, checkpointing, and ridge regularization sweep).

## What this project demonstrates
- **From-scratch linear regression training loop in PyTorch** (explicit w, b, forward pass, loss.backward(), manual GD updates)
- **Train/Val/Test workflow** with reproducible splits (fixed split & pytorch training seeds)
- Metrics: **MSE, RMSE, R²**
- **Best validation checkpointing**
- **Ridge (L2) regularization sweep** to select λ by validation performance
- Simple plotting of learning curves

## Under the hood (PyTorch from scratch)
This project intentionally avoids high-level training helpers to show fundamentals. I implemented:
- **Explicit parameters**: `w` and `b` as tensors (no `nn.Linear`)
- **Forward pass**: predictions computed as `y_hat = X @ w + b`
- **Loss**: mean squared error (MSE)
- **Backward pass**: gradients computed via `loss.backward()` and **manual gradient descent updates** in a `torch.no_grad()` block
- **Training loop mechanics**: gradient zeroing, metric logging (MSE/RMSE/R²), and saving the **best validation checkpoint**

## Quickstart
From the repo root, you can run everything with two scripts (no need to go into `src/`):

```bash
pip install -r requirements.txt
python src.train.py
python src.eval.py
```

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
- Test MSE around **~0.63**
- Test RMSE aroud **~0.79**
- Test R² around **~0.52**
- Ridge sweep found best near **λ = 0.01**

## Setup (optional but recommended): virtual environment
If you prefer an isolated environment:

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
