import numpy as np


def _delta_pareto(F_old, F_new):
    domination_matrix = calc_domination_matrix(F_new, F_old)

    keep_new_F = np.where(np.logical_not(np.any(domination_matrix < 0, axis=1)))[0]
    keep_old_F = np.where(np.logical_not(np.any(domination_matrix > 0, axis=0)))[0]

    return keep_new_F, keep_old_F


def _update_pareto(x_old, x_new, F_old, F_new):
    # Pareto on the new guys
    x_new = x_new.astype(np.float)

    # Remove duplicate of new guys
    unique_i = np.unique(x_new, axis=0, return_index=True)[1]
    x_new = x_new[unique_i]
    F_new = F_new[unique_i]

    # Calculate pareto of new guys
    pareto_i = self.nds.do(F_new)[0]
    pareto_F = F_new[pareto_i]
    pareto_x = x_new[pareto_i]

    new_pareto_i, old_pareto_i = self._delta_pareto(F_old, pareto_F)

    out_pareto_X = np.concatenate((pareto_x[new_pareto_i], F_old[old_pareto_i]))
    out_pareto_F = np.concatenate((pareto_F[new_pareto_i], x_old[old_pareto_i]))

    return out_pareto_X, out_pareto_F


def calc_domination_matrix(F_new, F_old, epsilon=0.0):
    # look at the obj for dom
    n = F_new.shape[0]
    m = F_old.shape[0]

    L = np.repeat(F_new, m, axis=0)
    R = np.tile(F_old, (n, 1))

    smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
    larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

    M = (
        np.logical_and(smaller, np.logical_not(larger)) * 1
        + np.logical_and(larger, np.logical_not(smaller)) * -1
    )

    return M
