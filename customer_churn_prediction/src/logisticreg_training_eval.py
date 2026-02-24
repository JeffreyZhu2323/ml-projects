from data_loading import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import json
import matplotlib.pyplot as plt

X_train,y_train,X_val,y_val,X_test,y_test = load_data()

clf = LogisticRegression(
        max_iter=3000, 
        random_state=TRAIN_SEED,
        C = 1,
        class_weight = None,
        penalty = "l1",
        solver =  "saga"
    )

clf.fit(X_train, y_train)
val_probs = clf.predict_proba(X_val)[:,1]
test_probs = clf.predict_proba(X_test)[:,1]

# Brier score (lower is better; perfectly calibrated model ~ 0)
brier = brier_score_loss(y_test, test_probs)

thr_05 = 0.5
top_05_threshold = np.quantile(test_probs,0.95)
top_10_threshold = np.quantile(test_probs,0.9)
top_20_threshold = np.quantile(test_probs,0.8)
top_30_threshold = np.quantile(test_probs,0.7)
best_f1_threshold = get_best_f1_thr(y_val,val_probs) 

metrics_path = BASE_DIR/"results/performance_metrics.json"
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f) 

metrics.update({"logistic_regression_tuned": {
"roc_auc": roc_auc_score(y_test,test_probs),
"pr_auc": average_precision_score(y_test,test_probs),
"brier_score": brier,
"threshold_results": {
}
}
}) 
metrics["logistic_regression_tuned"]["threshold_results"]["thr_0.5"] = eval_at_threshold(y_test,test_probs,thr_05)
metrics["logistic_regression_tuned"]["threshold_results"]["top_05_percent"] = eval_at_threshold(y_test,test_probs,top_05_threshold)
metrics["logistic_regression_tuned"]["threshold_results"]["top_20_percent"] = eval_at_threshold(y_test,test_probs,top_20_threshold)
metrics["logistic_regression_tuned"]["threshold_results"]["top_30_percent"] = eval_at_threshold(y_test,test_probs,top_30_threshold)
metrics["logistic_regression_tuned"]["threshold_results"]["top_10_percent"] = eval_at_threshold(y_test,test_probs,top_10_threshold)
metrics["logistic_regression_tuned"]["threshold_results"]["best_f1_score_threshold"] = eval_at_threshold(y_test,test_probs,best_f1_threshold)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved metrics to {metrics_path}")
coefficients = pd.Series(clf.coef_[0], index=X_train.columns).sort_values()
top_increasing = coefficients.sort_values(ascending=False)                        

top_increasing.to_csv(BASE_DIR/"results/variable_affect_churn.csv", header=["coefficients"])
print("Saved results/variable_affect_curn.csv")

# Calibration curve (reliability diagram)
reports_dir = BASE_DIR / "reports"
prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=10, strategy="quantile")

plt.figure(figsize=(6, 5))
plt.plot(prob_pred, prob_true, marker="o", linewidth=1, label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
plt.xlabel("Predicted probability")
plt.ylabel("Observed fraction of positives")
plt.title("Calibration Curve (Reliability Diagram) - Logistic Regression")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(reports_dir / "logistic_calibration_curve.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved calibration curve to: {reports_dir / 'logistic_calibration_curve.png'}")