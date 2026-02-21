from data_loading import *
from xgboost import XGBClassifier

X_train,y_train,X_val,y_val,X_test,y_test = load_data()

pos = y_train[y_train == 1].shape[0]
neg = y_train[y_train == 0].shape[0]
scale_pos = pos/neg
print(pos, neg)
xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SPLIT_SEED,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos,
    )

    # 4) Train (using validation set for monitoring only)
xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

    # 5) Predict probabilities (NOT hard labels)
val_proba = xgb.predict_proba(X_val)[:, 1]
test_proba = xgb.predict_proba(X_test)[:, 1]