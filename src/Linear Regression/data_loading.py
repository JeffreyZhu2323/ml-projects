from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch
from config import *

def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    features = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup"]
    X = df[features].values          
    y = df['MedHouseVal'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SPLIT_SIZE, random_state=SPLIT_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=SPLIT_SIZE, random_state=SPLIT_SEED
    )
    X_train = torch.tensor(X_train, dtype = torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype = torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype = torch.float32)
    y_test = torch.tensor(y_test, dtype = torch.float32)
    mean = X_train.mean(dim=0,keepdim = True)
    std  = X_train.std(dim=0,keepdim = True)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    X_val = (X_val - mean) / std
    return X_train,y_train, X_test, y_test, X_val, y_val, mean, std
