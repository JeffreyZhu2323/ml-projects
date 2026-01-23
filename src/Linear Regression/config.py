from pathlib import Path

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

EPOCHS = 1500

LR = 0.01

SPLIT_SIZE = 0.2

SPLIT_SEED = 42

BASE_DIR = Path(__file__).resolve().parent
SAVE_PATH = BASE_DIR / "model_lr.pt"
