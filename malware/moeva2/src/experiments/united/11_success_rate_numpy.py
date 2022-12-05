import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.objective_calculator import ObjectiveCalculator
from src.experiments.united.utils import get_constraints_from_str
from src.utils import in_out, filter_initial_states

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def run(config):
    Path(config["paths"]["objectives"]).parent.mkdir(parents=True, exist_ok=True)

    # ----- Load and create necessary objects

    results = np.load(config["paths"]["attack_results"])
    x_initial = np.load(config["paths"]["x_candidates"])
    x_initial = filter_initial_states(
        x_initial, config["initial_state_offset"], config["n_initial_state"]
    )

    constraints = get_constraints_from_str(config["project_name"])(
        config["paths"]["features"],
        config["paths"]["constraints"],
    )

    classifier = Classifier(load_model(config["paths"]["model"]))
    scaler = joblib.load(config["paths"]["min_max_scaler"])

    objective_calc = ObjectiveCalculator(
        classifier,
        constraints,
        minimize_class=1,
        thresholds=config["thresholds"],
        min_max_scaler=scaler,
        ml_scaler=scaler,
    )

    if len(results.shape) == 2:
        results = results[:, np.newaxis, :]

    success_rates = objective_calc.success_rate_3d(x_initial, results)

    columns = ["o{}".format(i + 1) for i in range(success_rates.shape[0])]
    success_rate_df = pd.DataFrame(
        success_rates.reshape([1, -1]),
        columns=columns,
    )
    success_rate_df.to_csv(config["paths"]["objectives"], index=False)
    print(success_rate_df)


if __name__ == "__main__":
    config = in_out.get_parameters()
    run(config)
