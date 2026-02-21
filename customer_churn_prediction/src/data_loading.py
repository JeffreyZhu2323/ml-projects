import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from src.config import *

def load_data():
    csv_path = BASE_DIR / "data" / "Telco_customer_churn.csv"
    xlsx_path = BASE_DIR / "data" / "Telco_customer_churn.xlsx"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(...)

    features = [ 'Gender', 'Senior Citizen',
       'Partner', 'Dependents', 'Tenure Months', 'Phone Service',
       'Multiple Lines', 'Internet Service', 'Online Security',
       'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
       'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method',
       'Monthly Charges', 'Total Charges']
    
    X = df[features].copy()
    y = df["Churn Value"].astype("int8")

    X["Total Charges"] = pd.to_numeric(X["Total Charges"], errors= "coerce")
    mask = (X["Total Charges"].isna()) & (X["Tenure Months"] == 0)
    X.loc[mask, "Total Charges"] = 0
    X["Monthly Charges"] = pd.to_numeric(X["Monthly Charges"],errors = "coerce")
    mapping = {"No": 0,"Yes": 1,"Male":0,"Female":1, "No internet service":0, "No phone service": 0,0: 0, 1: 1}

    binary_cols = [
    "Senior Citizen","Partner","Dependents","Phone Service","Multiple Lines",
    "Online Security","Online Backup","Device Protection","Tech Support",
    "Streaming TV","Streaming Movies","Paperless Billing","Gender"
    ]

    for c in binary_cols:
        X[c] = X[c].map(mapping)
    X[binary_cols] = X[binary_cols].astype("int8")
    X = pd.get_dummies(X, columns=["Internet Service","Contract","Payment Method"],drop_first=True,dtype=np.int8)
 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE, random_state=SPLIT_SEED, stratify = y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=TEST_SPLIT_SIZE, random_state=SPLIT_SEED, stratify = y_train
    )

    continuous_cols = ["Tenure Months", "Monthly Charges", "Total Charges"]
    mean = X_train[continuous_cols].mean()
    std = X_train[continuous_cols].std(ddof=0)
    X_train[continuous_cols] = (X_train[continuous_cols] - mean) / std
    X_val[continuous_cols] = (X_val[continuous_cols] - mean) / std
    X_test[continuous_cols] = (X_test[continuous_cols] - mean) / std

    print("Cleaned and split data")
    return X_train,y_train,X_val,y_val,X_test,y_test

