import torch
from config import *
from data_loading import *

_, _, X_test, y_test,_,_ = load_data()
checkpoint = torch.load(SAVE_PATH)
w,b = checkpoint["w"], checkpoint["b"]
with torch.no_grad():
    preds = X_test @ w + b
    mse = mse_loss(preds, y_test).item()
    rmse = rmse_loss(preds, y_test).item()
    r2_loss = r2(preds,y_test).item()

print("MSE:", mse)
print("RMSE:",rmse)
print("R2:",r2_loss)