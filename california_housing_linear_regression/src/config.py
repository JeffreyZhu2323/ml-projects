from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

def rmse_loss(y_pred,  y_true):
    return torch.sqrt(mse_loss(y_pred, y_true))

def r2(y_pred, y_true, eps: float = 1e-12):
    res = torch.sum((y_true - y_pred) ** 2)
    tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1.0 - res / (tot + eps)

EPOCHS = 2000

SPLINE_COL_INDICES = [1, 2]  # HouseAge, AveRooms only
SPLINE_NAMES = ["HouseAge", "AveRooms"]
N_SPLINE_KNOTS = 3  # interior knots per feature

LR = 0.3

TEST_SPLIT_SIZE = 0.2
VAL_SPLIT_SIZE = 0.2

SPLIT_SEED = 42

TORCH_SEED = 42

L2_LAMBDA = 10 ** -5
lambdas = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
  

BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_PATH = BASE_DIR / "checkpoints" / "model_lr"

def quantile_plot(x, y, by=None, bins=10, by_bins=3, y_fn=np.mean):
    assert len(x) == len(y)

    def qp_data(x, y):
        fac = np.searchsorted(np.quantile(x, q=[i / bins for i in range(1, bins)]), x)
        ufac = np.unique(fac)
        qx = np.array([np.mean(x[fac == f]) for f in ufac])
        qy = np.array([y_fn(y[fac == f]) for f in ufac])
        return qx, qy

    qx, qy = qp_data(x, y)
    if by is None:
        plt.plot(qx, qy, "-o")
    else:
        assert len(x) == len(by)
        plt.plot(qx, qy, "-o", label="ALL", color="lightgrey")
        by_fac = np.searchsorted(np.quantile(by, q=[i / by_bins for i in range(1, by_bins)]), by)
        by_ufac = np.unique(by_fac)
        for i, f in enumerate(np.unique(by_ufac)):
            mask = by_fac == f
            nm = f"{i}) {min(by[mask]):.2f} / {max(by[mask]):.2f}"
            qx, qy = qp_data(x[mask], y[mask])
            plt.plot(qx, qy, "-o", label=nm)
        plt.legend()
