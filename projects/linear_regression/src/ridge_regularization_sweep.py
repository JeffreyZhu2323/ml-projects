from data_loading import load_data
import torch
from config import *
import json

#load data
X_train,y_train,_,_,X_val,y_val = load_data()

#make results reproducible
torch.manual_seed(TORCH_SEED)

#instantiate parameters
w = torch.randn(6, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

w_best = torch.empty(6,1)
b_best = torch.empty(1)
best_loss = float("inf")
global_w_best = None
global_b_best = None
global_best = float('inf')
best_lambda = None
best = {}

def model(X,w,b):
    return X @ w + b

#training cycle
for L2 in lambdas:
    torch.manual_seed(TORCH_SEED)
    #instantiate parameters
    w = torch.randn(6, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    w_best = torch.empty(6,1)
    b_best = torch.empty(1)
    best_loss = float("inf")
    for epoch in range(EPOCHS):
        #forward + back prop
        train_pred = model(X_train,w,b)
        l2_penalty = (w ** 2).sum() * L2
        train_mse = mse_loss(train_pred, y_train) 
        loss = train_mse + l2_penalty
        loss.backward()

        with torch.no_grad():
            #evaluate validation set and save model weights if performs best
            val_preds = model(X_val,w,b)
            val_mse = mse_loss(val_preds, y_val).item()
            if val_mse < best_loss:
                w_best = w.detach().clone()
                b_best = b.detach().clone()
                best_loss = val_mse
            
            #update and reset weights
            w -= LR * w.grad
            b -= LR * b.grad
            w.grad.zero_()
            b.grad.zero_()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train MSE: {train_mse.item()}, Val MSE: {val_mse}")

    #after training cycle finished for 1 lambda value update the global bests
    best[L2] = best_loss
    if best_loss < global_best:
        global_b_best = b_best.clone()
        global_w_best = w_best.clone()
        best_lambda = L2
        global_best = best_loss

print(best)
print("Best Lambda Value:",best_lambda)

#save results of all lambda values
results_json = {str(k): float(v) for k, v in best.items()}
with open(BASE_DIR / "results" / "ridge_sweep_results.json", "w") as f:
    json.dump(results_json, f, indent=2)

print("Saved to ridge_sweep_results.json")