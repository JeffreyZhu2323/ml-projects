from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import precision_recall_curve

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_SPLIT_SIZE = 0.2
TRAIN_SEED = 42
SPLIT_SEED = 42

def eval_at_threshold(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return  {
        "threshold": float(thr),
        "target_rate": float(pred.mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }

def get_best_f1_thr(y_true, proba):

    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = np.argmax(f1)

    return thresholds[best_index]
    
