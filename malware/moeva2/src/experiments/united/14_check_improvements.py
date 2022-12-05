import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.attacks.moeva2.utils import get_scaler_from_norm
from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.objective_calculator import ObjectiveCalculator
from src.examples.lcld.lcld_constraints import LcldConstraints
from src.utils import Pickler, in_out

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

config = in_out.get_parameters()
from src.utils.in_out import load_model

AVOID_ZERO = 0.00000001


def run():
    Path(config["paths"]["objectives"]).parent.mkdir(parents=True, exist_ok=True)

    # ----- Load and create necessary objects

    efficient_results = Pickler.load_from_file(config["paths"]["attack_results"])

    histories = [
        [g["F"].tolist() for i, g in enumerate(r.history) if i > 0]
        for r in efficient_results
    ]
    histories = np.array(histories)[:,:100,:,:]

    # Objective scalers (Compute only once)

    f2_scaler = get_scaler_from_norm(
        config["norm"], efficient_results[0].initial_state.shape[0]
    )

    shape = histories[..., 0].shape
    histories[..., 1] = f2_scaler.inverse_transform(histories[..., 1].reshape(-1, 1)).reshape(shape)

    misclassified = (histories[..., 0] < config["thresholds"]["f1"]).astype(np.float)
    distance = (histories[..., 1] <= config["thresholds"]["f2"]).astype(np.float)
    constraints = (histories[..., 2] <= 0).astype(np.float)

    alles = misclassified * distance * constraints

    for i, e in enumerate([misclassified, distance, constraints, alles]):
        all_sum = e.sum(axis=2)
        all_avg = np.mean(all_sum, axis=0)
        plt.plot(all_avg, label=f"{i}")

    # all_avg = np.convolve(all_avg, np.ones(10), 'valid') / 10
    print(alles.shape)
    success_rate = (np.sum(np.sum(alles, axis=2), axis=1) > 0).astype(np.float)
    print(f"Success rate {success_rate}")

    # plt.ylim(bottom=-1.0)
    # plt.plot(working[:, 0], label=f"{0}")

    # plot thresholds
    # for key in config["thresholds"]:
    #     plt.plot(
    #         np.full(working.shape[0], config["thresholds"][key]),
    #         label=f"Threshold {key}",
    #     )

    plt.yscale("linear")
    plt.legend()
    plt.savefig("improvements_improvements_malware_rf_rnsga3_05.pdf")



if __name__ == "__main__":
    run()
