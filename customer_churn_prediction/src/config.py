from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_SPLIT_SIZE = 0.2

SPLIT_SEED = 42

def eval_at_threshold(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "threshold": float(thr),
        "target_rate": float(pred.mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }

    
