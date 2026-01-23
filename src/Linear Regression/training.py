from data_loading import load_data
import torch
from config import *
import numpy as np

X_train,y_train,_,_,X_val,y_val, mean, std = load_data()

torch.manual_seed(42)
w = torch.randn(6, 1, requires_grad=True)
w_best = torch.empty(6,1)
b = torch.zeros(1, requires_grad=True)
b_best = torch.empty(1)
best_loss = np.inf

def model(X):
    return X @ w + b
for epoch in range(EPOCHS):
    y_pred = model(X_train)
    loss = mse_loss(y_pred, y_train)
    loss.backward()
    with torch.no_grad():
        w -= LR * w.grad
        b -= LR * b.grad
    w.grad.zero_()
    b.grad.zero_()
    with torch.no_grad():
        val_preds = model(X_val)
        val_loss = mse_loss(val_preds, y_val)
        if val_loss < best_loss:
            w_best = w.detach()
            b_best = b.detach()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Train MSE: {loss.item():.4f}, Val MSE: {val_loss}")

torch.save(
    {
        "w": w_best,
        "b": b_best,
        "mean":mean,
        "std":std
    },
    SAVE_PATH
)
