from data_loading import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import json


X_train,y_train,X_val,y_val,X_test,y_test = load_data()

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train, y_train)

val_probs = clf.predict_proba(X_val)[:,1]
precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
best_index = np.argmax(f1)
best_f1_threshold = thresholds[best_index]

test_probs = clf.predict_proba(X_test)[:,1]
metrics_path = BASE_DIR/"results/baseline_metrics"
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)   
metrics["threshold_results"]["best_f1_score_threshold"] = eval_at_threshold(y_test,test_probs,best_f1_threshold)
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics on new threshold:{best_f1_threshold} to /results/baseline_metrics")

coefficients = pd.Series(clf.coef_[0], index=X_train.columns).sort_values()
top_increasing = coefficients.sort_values(ascending=False)                        

top_increasing.to_csv(BASE_DIR/"results/variable_affect_churn.csv", header=["coefficients"])
print("Saved results/variable_affect_curn.csv")
