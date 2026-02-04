import torch
from src.config import *
from src.data_loading import *
import json

def main():
    results = {}
    _, _, X_test, y_test,_,_ = load_data()
    checkpoint = torch.load(SAVE_PATH)
    w,b = checkpoint["w"], checkpoint["b"]
    with torch.no_grad():
        preds = X_test @ w + b
        mse = mse_loss(preds, y_test).item()
        rmse = rmse_loss(preds, y_test).item()
        r2_loss = r2(preds,y_test).item()
    results["MSE"] = mse
    results["RMSE"] = rmse
    results["R^2"] = r2_loss
    print("MSE:", mse)
    print("RMSE:",rmse)
    print("R2:",r2_loss)

    #save results of all lambda values
    results_json = {str(k): float(v) for k, v in results.items()}
    with open(BASE_DIR / "results" / "test_set_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print("Saved to test_set_results.json")