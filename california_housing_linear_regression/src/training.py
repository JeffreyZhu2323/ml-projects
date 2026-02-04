from src.data_loading import load_data
import torch
from src.config import *
import pandas as pd

def main():
    #load data
    X_train,y_train,_,_,X_val,y_val = load_data()

    #make results reproducible
    torch.manual_seed(TORCH_SEED)

    #instantiate parameters
    w = torch.randn(6, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    w_best = torch.empty(X_train.shape[1],1)
    b_best = torch.empty(1)
    best_loss = float("inf")
    loss_graph = pd.DataFrame(columns=["epoch","train_mse","val_mse","train_rmse","val_rmse","train_r2","val_r2"])

    def model(X,w,b):
        return X @ w + b

    for epoch in range(1,EPOCHS + 1):
        #forward + back prop
        train_pred = model(X_train,w,b)
        l2_penalty = (w ** 2).sum() * L2_LAMBDA
        train_mse = mse_loss(train_pred, y_train) 
        loss = train_mse + l2_penalty
        loss.backward()
        train_rmse = rmse_loss(train_pred, y_train).item()
        train_r2 = r2(train_pred,y_train).item()

        #evaluate validation set and save model weights if performs best
        with torch.no_grad():
            val_preds = model(X_val,w,b)
            val_mse = mse_loss(val_preds, y_val).item()
            val_rmse = rmse_loss(val_preds,y_val).item()
            val_r2 = r2(val_preds,y_val).item()
            if val_mse < best_loss:
                w_best = w.detach().clone()
                b_best = b.detach().clone()
                best_loss = val_mse
            
            #update and reset weights
            w -= LR * w.grad
            b -= LR * b.grad
            w.grad.zero_()
            b.grad.zero_()

        #log loss curves
        loss_graph.loc[len(loss_graph)] = {"epoch":epoch,"train_mse":train_mse.item(),"val_mse":val_mse,"train_rmse":train_rmse,"val_rmse":val_rmse,"train_r2":train_r2,"val_r2":val_r2}
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train MSE: {train_mse.item()}, Val MSE: {val_mse}")

    #save loss graphs and model
    loss_graph.to_csv(BASE_DIR / "results" / "loss_graph.csv")

    torch.save(
        {
            "w": w_best,
            "b": b_best,
        },
        SAVE_PATH
    )
    print("Saved model!")
