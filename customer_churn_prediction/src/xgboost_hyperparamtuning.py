from data_loading import *
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
import json

X_train, y_train, X_val,y_val,_,_ = load_data()

pos = y_train[y_train == 1].shape[0]
neg = y_train[y_train == 0].shape[0]
scale_pos = neg / pos

xgb_base = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    eval_metric="logloss",   
    n_estimators=600,        
    random_state=42,
    n_jobs=1,               
)

xgb_param_grid = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [1.0, 3.0],
    "scale_pos_weight": [1.0, scale_pos], 
}

gridxg = GridSearchCV(
    estimator=xgb_base,
    param_grid=xgb_param_grid,
    scoring={
        "roc_auc": "roc_auc",
        "ap": "average_precision",   
        "accuracy": "accuracy",
    },
    refit="ap",                 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,                      
    verbose=2,
    return_train_score=True,
    error_score="raise",
)

#running hyperparameter tuning
gridxg.fit(X_train, y_train)

print("\nBest params (CV):")
print(gridxg.best_params_)
print("Best CV PR-AUC:", gridxg.best_score_)

cv_resultsxg = pd.DataFrame(gridxg.cv_results_)
show_cols = [
    "rank_test_ap",
    "mean_test_ap",
    "std_test_ap",
    "mean_test_roc_auc",
    "mean_test_accuracy",
    "mean_train_ap",   
]

param_colsxg = [c for c in cv_resultsxg.columns if c.startswith("param_")]
topxg = cv_resultsxg[show_cols + param_colsxg].sort_values("rank_test_ap").head(10)
print(topxg.to_string(index=False))

summaryxg = {
    "model": "xgboost",
    "refit_metric": "ap",
    "best_params": gridxg.best_params_,
    "best_index": int(gridxg.best_index_),
    "best_prauc_score": float(gridxg.best_score_),
}

file_pathlr = BASE_DIR / "results" / "best_xgboost_params"
with open(file_pathlr, 'w') as json_file:
    json.dump(summaryxg, json_file, indent=4)
