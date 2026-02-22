from data_loading import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import json

X_train, y_train, X_val,y_val,_,_ = load_data()

lr_estimator = LogisticRegression(
    max_iter=3000,
    random_state=42,
)

lr_param_grid = [
    {
        "solver": ["lbfgs"],
        "penalty": ["l2"],
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"],
    },
    {
        "solver": ["liblinear"],
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"],
    },
    {
        "solver": ["saga"],
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"],
    },
    {
        "solver": ["saga"],
        "penalty": ["elasticnet"],
        "C": [0.01, 0.1, 1, 10],
        "l1_ratio": [0.2, 0.5, 0.8],
        "class_weight": [None, "balanced"],
    },
]

gridlr = GridSearchCV(
    estimator=lr_estimator,
    param_grid=lr_param_grid,
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
    error_score="raise"
)

gridlr.fit(X_train, y_train)

print("\nBest params Logistic Regression:")
print(gridlr.best_params_)
print("Best PR-AUC:", gridlr.best_score_)

show_cols = [
    "rank_test_ap",
    "mean_test_ap",
    "std_test_ap",
    "mean_test_roc_auc",
    "mean_test_accuracy",
    "mean_train_ap",   
]

cv_resultslr = pd.DataFrame(gridlr.cv_results_)
param_colslr = [c for c in cv_resultslr.columns if c.startswith("param_")]
toplr = cv_resultslr[show_cols + param_colslr].sort_values("rank_test_ap").head(10)
print(toplr.to_string(index=False))

summarylr= {
    "model": "Logistic Regression",
    "refit_metric": "ap",
    "best_params": gridlr.best_params_,
    "best_index": int(gridlr.best_index_),
    "best_prauc_score": float(gridlr.best_score_),
}

file_pathlr = BASE_DIR / "results" / "best_logisticreg_params"
with open(file_pathlr, 'w') as json_file:
    json.dump(summarylr, json_file, indent=4)