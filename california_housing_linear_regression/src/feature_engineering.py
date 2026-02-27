from data_loading import load_data
from config import quantile_plot
import matplotlib.pyplot as plt
import torch


X_train, y_train, _, _, X_val, y_val = load_data()


def make_features(X: torch.Tensor, return_feature_names: bool = False):
    """
    Given a base feature matrix X with columns:
        [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup]
    return an expanded feature matrix: raw features plus cutoff/hinge features.
    """
    med_inc = X[:, 0:1]
    house_age = X[:, 1:2]
    ave_rooms = X[:, 2:3]
    ave_bedrms = X[:, 3:4]
    population = X[:, 4:5]
    ave_occup = X[:, 5:6]

    feats = []
    names = []

    # Raw base features
    feats.extend([med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup])
    names.extend(["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"])

    # Cutoff features (floor for HouseAge/AveRooms, hinge for Population)
    feats.append(torch.clamp_min(house_age, 1.1))
    names.append("HouseAge_high")
    feats.append(torch.clamp_min(ave_rooms, 0.5))
    names.append("AveRooms_high")
    feats.append(torch.clamp_min(population - 0.5, 0.0))
    names.append("Population_high")

    # AveOccup: hinge above 0.1 (extra negative slope for high occupancy)
    avocc_high = torch.clamp_min(ave_occup - 0.1, 0.0)
    feats.append(avocc_high)
    names.append("AveOcc_high")

    X_feat = torch.cat(feats, dim=1)
    if return_feature_names:
        return X_feat, names
    return X_feat


def show_candidate_feature_quantile_plots():
    """Display quantile plots for all engineered candidate features."""
    X_feat, feat_names = make_features(X_train, return_feature_names=True)
    X_np = X_feat.numpy()
    y_np = y_train.numpy().ravel()

    for idx, name in enumerate(feat_names):
        plt.figure()
        quantile_plot(X_np[:, idx], y_np)
        plt.xlabel(name)
        plt.ylabel("MedHouseVal")
        plt.title(f"Quantile plot of {name} vs MedHouseVal")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Base features:
    # All engineered candidates:
    show_candidate_feature_quantile_plots()
