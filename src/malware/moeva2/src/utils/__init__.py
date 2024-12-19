from functools import wraps
from time import time

import numpy as np
from sklearn.metrics import matthews_corrcoef


def get_data_by_month(data, a_month):
    month_df = data[data["issue_d"] == a_month]
    a_y = month_df.pop("charged_off").to_numpy()
    a_X = month_df.to_numpy()
    return a_X, a_y


def filter_initial_states(x, start, size):
    if size > -1:
        return x[start : start + size]
    else:
        return x


def random_sample_hyperball(n, d):
    u = np.random.normal(0, 1, (d + 2) * n).reshape(n, d + 2)
    norm = np.linalg.norm(u, axis=1)
    u = u / norm.reshape(-1, 1)
    x = u[:, 0:d]
    return x


def sample_in_norm(n_samples, d, eps, norm):
    if norm in ["2", 2]:
        x_perturbation = random_sample_hyperball(n_samples, d) * eps

    elif norm in ["inf", np.inf]:
        x_perturbation = np.random.random((n_samples, d)) * 2 - 1
        x_perturbation = x_perturbation * eps

    else:
        raise NotImplementedError

    return x_perturbation


def find_best_threshold(y_test, y_proba, metric=matthews_corrcoef, step=0.01):
    nb_steps = int(1 / step)
    metric_values = [
        metric(y_test, (y_proba >= t / nb_steps).astype(int)) for t in range(nb_steps)
    ]
    best_i = np.argmax(metric_values)
    best_t = best_i / nb_steps
    best_metric = metric_values[best_i]

    return best_t, best_metric


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__!r} args:[{args!r}, {kw!r}] took: {te - ts:2.4f} sec")
        return result

    return wrap
