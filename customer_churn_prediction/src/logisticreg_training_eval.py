from data_loading import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import json

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

thr_05 = 0.5
top_10_threshold = np.quantile(test_probs,0.9)
best_f1_threshold = get_best_f1_thr(y_val,val_probs) 

metrics_path = BASE_DIR/"results/performance_metrics"
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f) 

metrics.update({"logistic_regression_tuned": {
"roc_auc": roc_auc_score(y_test,test_probs),
"pr_auc": average_precision_score(y_test,test_probs),
"threshold_results": {
}
}
}) 
metrics["logistic_regression_tuned"]["threshold_results"]["thr_0.5"] = eval_at_threshold(y_test,test_probs,thr_05)
metrics["logistic_regression_tuned"]["threshold_results"]["top_10_percent"] = eval_at_threshold(y_test,test_probs,top_10_threshold)
metrics["logistic_regression_tuned"]["threshold_results"]["best_f1_score_threshold"] = eval_at_threshold(y_test,test_probs,best_f1_threshold)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved metrics to {metrics_path}")
coefficients = pd.Series(clf.coef_[0], index=X_train.columns).sort_values()
top_increasing = coefficients.sort_values(ascending=False)                        

top_increasing.to_csv(BASE_DIR/"results/variable_affect_churn.csv", header=["coefficients"])
print("Saved results/variable_affect_curn.csv")