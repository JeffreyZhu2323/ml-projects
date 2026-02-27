import pandas as pd
import matplotlib.pyplot as plt
from config import * 

loss_graph = pd.read_csv(BASE_DIR / "results" / "loss_graph.csv")
(BASE_DIR / "reports").mkdir(exist_ok=True)

# Save RMSE
plt.figure()
plt.plot(loss_graph["train_rmse"], label="Train")
plt.plot(loss_graph["val_rmse"], label="Val")
plt.title("RMSE Curve")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.ylim(0, 2)
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "reports" / "rmse_curve.png")
plt.close()

# Save MSE
plt.figure()
plt.plot(loss_graph["train_mse"], label="Train")
plt.plot(loss_graph["val_mse"], label="Val")
plt.title("MSE Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "reports" / "mse_curve.png")
plt.close()

# Save R^2
plt.figure()
plt.plot(loss_graph["train_r2"], label="Train")
plt.plot(loss_graph["val_r2"], label="Val")
plt.title("R^2 Curve")
plt.xlabel("Epoch")
plt.ylabel("R^2")
plt.ylim(-1, 1)
plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "reports" / "r2_curve.png")
plt.close()


