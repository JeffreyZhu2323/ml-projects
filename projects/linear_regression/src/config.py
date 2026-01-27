from pathlib import Path
import torch
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

def rmse_loss(y_pred,  y_true):
    return torch.sqrt(mse_loss(y_pred, y_true))

def r2(y_pred, y_true, eps: float = 1e-12):
    res = torch.sum((y_true - y_pred) ** 2)
    tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1.0 - res / (tot + eps)

EPOCHS = 2000

LR = 0.01

TEST_SPLIT_SIZE = 0.2
VAL_SPLIT_SIZE = 0.2

SPLIT_SEED = 42

TORCH_SEED = 42

L2_LAMBDA = 0.01
lambdas = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
  

BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_PATH = BASE_DIR / "model" / "model_lr"
