from typing import List

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.attacks.result_process import EfficientResult, HistoryResult

ONE_HOT_ENCODE_KEY = "ohe"


def get_scaler_from_norm(norm, nb_features):

    scaler = MinMaxScaler(feature_range=(0, 1))

    if norm in [2, "2"]:
        scaler.fit([[0], [np.sqrt(nb_features)]])
    elif norm in [np.inf, "inf"]:
        scaler.fit([[0], [1]])
    else:
        raise NotImplementedError

    return scaler


def get_ohe_masks(type_mask):
    seen_key = []
    one_hot_masks = []

    for i, e_type in enumerate(type_mask):
        if e_type.startswith(ONE_HOT_ENCODE_KEY):
            if e_type in seen_key:
                index = seen_key.index(e_type)
                one_hot_masks[index].append(i)
            else:
                seen_key.append(e_type)
                one_hot_masks.append([i])

    one_hot_masks = [np.array(e) for e in one_hot_masks]

    return one_hot_masks


def get_one_hot_encoding_constraints(type_mask, x):

    one_hot_masks = get_ohe_masks(type_mask)
    if len(one_hot_masks) == 0:
        return np.zeros(x.shape[0])

    one_hot_values = np.column_stack(
        [np.sum(x[:, one_hot_mask], axis=1) for one_hot_mask in one_hot_masks]
    )
    one_hot_distance = np.abs(1 - one_hot_values)
    one_hot_distance = np.sum(one_hot_distance, axis=1)
    return one_hot_distance


def results_to_numpy_results(results: List[EfficientResult], encoder):

    initial_states = [result.initial_state for result in results]
    pops_x = [
        np.array([ind.X.astype(np.float64) for ind in result.pop]) for result in results
    ]
    # Convert to ML representation
    pops_x_f = [
        encoder.genetic_to_ml(pops_x[i], initial_states[i]) for i in range(len(results))
    ]
    return np.array(pops_x_f)


def results_to_history(results: List[HistoryResult]):

    histories = [
        [g.tolist() for i, g in enumerate(r.history) if i > 0] for r in results
    ]
    histories = np.array(histories)
    return histories
