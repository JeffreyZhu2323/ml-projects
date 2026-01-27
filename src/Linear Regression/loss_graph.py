import pandas as pd
import matplotlib.pyplot as plt

loss_graph = pd.read_csv("loss_graph.csv")
# --- RMSE ---
plt.figure()
plt.plot(loss_graph["train_rmse"], label="Train")
plt.plot(loss_graph["val_rmse"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.legend()
plt.show()

# --- MSE ---
plt.figure()
plt.plot(loss_graph["train_mse"], label="Train")
plt.plot(loss_graph["val_mse"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()

# --- R^2 ---
plt.figure()
plt.plot(loss_graph["train_r2"], label="Train")
plt.plot(loss_graph["val_r2"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("R^2")
plt.ylim(-1, 1)  # optional, R^2 is usually <= 1
plt.legend()
plt.show()

