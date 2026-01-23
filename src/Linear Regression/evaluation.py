import torch
from config import *
from data_loading import *
_, _, X_test, y_test,_,_,mean,std = load_data()
checkpoint = torch.load(SAVE_PATH)
w,b = checkpoint["w"], checkpoint["b"]
with torch.no_grad():
    preds = X_test @ w + b
    loss = mse_loss(preds, y_test)

print("MSE:", loss.item())