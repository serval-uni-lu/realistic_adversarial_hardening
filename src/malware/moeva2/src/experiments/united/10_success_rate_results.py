import warnings
from pathlib import Path

import joblib
import pandas as pd

from src.attacks.moeva2.classifier import Classifier
from src.attacks.moeva2.objective_calculator import ObjectiveCalculator
from src.experiments.united.utils import get_constraints_from_str
from src.utils import Pickler, in_out
from src.utils.in_out import load_model

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

config = in_out.get_parameters()


def run():
    Path(config["paths"]["objectives"]).parent.mkdir(parents=True, exist_ok=True)

    # ----- Load and create necessary objects

    efficient_results = Pickler.load_from_file(config["paths"]["attack_results"])

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
    success_rates = objective_calc.success_rate_genetic(efficient_results)

    columns = ["o{}".format(i + 1) for i in range(success_rates.shape[0])]
    success_rate_df = pd.DataFrame(
        success_rates.reshape([1, -1]),
        columns=columns,
    )
    success_rate_df.to_csv(config["paths"]["objectives"], index=False)
    print(success_rate_df)


if __name__ == "__main__":
    run()
