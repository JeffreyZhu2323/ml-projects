from main_data_loading import *
from xgboost import XGBClassifier
from config import *
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
import json
import matplotlib.pyplot as plt


metrics_path = BASE_DIR/"results/performance_metrics.json"

with open(metrics_path, 'r') as file:
    metrics = json.load(file)

X_train,y_train,X_val,y_val,X_test,y_test = load_data()

pos = y_train[y_train == 1].shape[0]
neg = y_train[y_train == 0].shape[0]
scale_pos = neg / pos

xgb = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",   
        n_estimators=600, 
        early_stopping_rounds=50 ,      
        random_state=TRAIN_SEED,
        n_jobs=1,
        colsample_bytree = 0.8,
        learning_rate = 0.03,
        max_depth = 3,
        min_child_weight = 1,
        reg_lambda = 3.0,
        scale_pos_weight = 1.0,
        subsample = 0.8
    )

xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

val_probs = xgb.predict_proba(X_val)[:,1]
test_probs = xgb.predict_proba(X_test)[:,1]

brier = brier_score_loss(y_test, test_probs)

thr_05 = 0.5
top_05_threshold = np.quantile(test_probs,0.95)
top_10_threshold = np.quantile(test_probs,0.9)
top_20_threshold = np.quantile(test_probs,0.8)
top_30_threshold = np.quantile(test_probs,0.7)
best_f1_threshold = get_best_f1_thr(y_val,val_probs) 

with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f) 

metrics.update({"xgboost_tuned": {
"roc_auc": roc_auc_score(y_test,test_probs),
"pr_auc": average_precision_score(y_test,test_probs),
"brier_score": brier,
"threshold_results": {
}
}
}) 
metrics["xgboost_tuned"]["threshold_results"]["thr_0.5"] = eval_at_threshold(y_test,test_probs,thr_05)
metrics["xgboost_tuned"]["threshold_results"]["top_05_percent"] = eval_at_threshold(y_test,test_probs,top_05_threshold)
metrics["xgboost_tuned"]["threshold_results"]["top_10_percent"] = eval_at_threshold(y_test,test_probs,top_10_threshold)
metrics["xgboost_tuned"]["threshold_results"]["top_20_percent"] = eval_at_threshold(y_test,test_probs,top_20_threshold)
metrics["xgboost_tuned"]["threshold_results"]["top_30_percent"] = eval_at_threshold(y_test,test_probs,top_30_threshold)
metrics["xgboost_tuned"]["threshold_results"]["best_f1_score_threshold"] = eval_at_threshold(y_test,test_probs,best_f1_threshold)


with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
reports_dir = BASE_DIR / "reports" / "figures"
print(f"Saved metrics to {metrics_path}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, test_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(reports_dir / "xgboost_roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, test_probs)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f"PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(reports_dir / "xgboost_pr_curve.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved ROC curve to: {reports_dir / 'xgboost_roc_curve.png'}")
print(f"Saved PR curve to: {reports_dir / 'xgboost_pr_curve.png'}")

# Calibration curve (reliability diagram)
prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=10, strategy="quantile")

plt.figure(figsize=(6, 5))
plt.plot(prob_pred, prob_true, marker="o", linewidth=1, label="XGBoost")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
plt.xlabel("Predicted probability")
plt.ylabel("Observed fraction of positives")
plt.title("Calibration Curve (Reliability Diagram)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(reports_dir / "xgboost_calibration_curve.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved calibration curve to: {reports_dir / 'xgboost_calibration_curve.png'}")
