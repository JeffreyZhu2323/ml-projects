# California Housing Linear Regression (PyTorch)

Linear regression on the California Housing dataset in **PyTorch**: manual GD, train/val/test workflow, feature engineering, and ridge (L2) regularization.

## What this project demonstrates
- **From-scratch linear regression** in PyTorch (explicit `w`, `b`, `loss.backward()`, manual GD)
- **Train/val/test** with reproducible splits and train-only standardization
- **Feature engineering** guided by quantile plots: cutoff and hinge features; splines were tried and dropped (no validation gain)
- **Ridge (L2) sweep** over λ after feature engineering; best λ chosen by validation MSE
- Metrics: MSE, RMSE, R²; best-val checkpointing; learning-curve plots

## Feature engineering
- **Base features (6):** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup
- **Engineered (5):** HouseAge_high (floor 1.1), AveRooms_high (floor 0.5), Population_high (hinge 0.5), AveOcc_high (hinge 0.1). Cutoff/hinge choices were informed by **quantile plots** of each variable vs target; a **transformation grid** of candidates was evaluated on validation; **splines** were tested and removed.

## Quickstart
From this project's directory (`california_housing_linear_regression/`), install deps then run from `src/`:

```bash
pip install -r requirements.txt
cd src
python training.py
python eval.py
```

Optional: `python ridge_regularization_sweep.py` (sweeps λ), `python loss_visual.py` (writes plots to `reports/`).

## Dataset
- **Source:** `sklearn.datasets.fetch_california_housing`
- **Target:** MedHouseVal

## Project structure
| Path | Role |
|------|------|
| `src/config.py` | Hyperparameters, metric helpers, paths, `quantile_plot` |
| `src/data_loading.py` | Load data, splits, standardize with train stats |
| `src/feature_engineering.py` | `make_features()` (11 features), quantile-plot EDA |
| `src/training.py` | Train with GD, save best-val checkpoint (w, b, train mean/std) |
| `src/eval.py` | Load checkpoint, evaluate on test set |
| `src/ridge_regularization_sweep.py` | L2 λ sweep, writes `results/ridge_sweep_results.json` |
| `src/loss_visual.py` | Plot train/val MSE, RMSE, R² → `reports/*.png` |
| `results/` | `loss_graph.csv`, `ridge_sweep_results.json`, `test_set_results.json` |
| `reports/` | Learning-curve PNGs (MSE, RMSE, R²) |
| `checkpoints/` | Saved model (w, b, train_mean, train_std) |

## Results
- **Test (after feature engineering + ridge sweep):** MSE ~0.54, RMSE ~0.74, R² ~0.59
- **Before feature engineering (6 raw features):** MSE ~0.63, RMSE ~0.79, R² ~0.52. Feature engineering + ridge sweep improved test R² by ~0.07 (≈13% relative).
- **Ridge sweep:** best validation MSE at **λ = 0.01**

## Setup (optional)
```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
```bash
# macOS/Linux
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
