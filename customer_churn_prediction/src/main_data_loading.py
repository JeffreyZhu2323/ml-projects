from sklearn.model_selection import train_test_split
from config import *
from google.cloud import bigquery

def load_from_bigquery(project: str, dataset: str, table: str):
    client = bigquery.Client(project=project)
    sql = f"SELECT * FROM `{project}.{dataset}.{table}` ORDER BY CustomerID"
    return client.query(sql).to_dataframe()

def load_data():
 
    df = load_from_bigquery("customerchurn-488906", "telcocustomerchurn", "churn_features")
    X = df.drop(columns=["churn","CustomerID"])
    y = df["churn"].astype("int8")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE, random_state=SPLIT_SEED, stratify = y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=TEST_SPLIT_SIZE, random_state=SPLIT_SEED, stratify = y_train
    )

    continuous_cols = ["tenure_months", "monthly_charges", "total_charges"]
    mean = X_train[continuous_cols].mean()
    std = X_train[continuous_cols].std(ddof=0)
    X_train[continuous_cols] = (X_train[continuous_cols] - mean) / std
    X_val[continuous_cols] = (X_val[continuous_cols] - mean) / std
    X_test[continuous_cols] = (X_test[continuous_cols] - mean) / std

    print("Cleaned and split data")
    return X_train,y_train,X_val,y_val,X_test,y_test