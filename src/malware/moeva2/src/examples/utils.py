from itertools import combinations
from math import comb
import tensorflow as tf
import numpy as np


def constraints_augmented_np(
    x: np.ndarray, important_features: np.ndarray, features_mean: np.ndarray
):
    n_important_features = len(important_features)
    x_augmented = x[:, -comb(n_important_features, 2) :]

    x_index = 0
    constraints = []
    combi = combinations(range(len(important_features)), 2)
    for i1, i2 in combi:
        g = np.abs(
            x_augmented[:, x_index]
            - (
                np.logical_xor(
                    (x[:, int(important_features[i1])] >= features_mean[i1]),
                    (x[:, int(important_features[i2])] >= features_mean[i2]),
                ).astype(np.float64)
            )
        )
        constraints.append(g)
        x_index += 1

    return constraints


def constraints_augmented_tf(
    x: np.ndarray, important_features: np.ndarray, features_mean: np.ndarray
):
    n_important_features = len(important_features)
    x_augmented = x[:, -comb(n_important_features, 2) :]

    x_index = 0
    constraints = []
    combi = combinations(range(len(important_features)), 2)
    for i1, i2 in combi:

        g = tf.math.abs(
            x_augmented[:, x_index]
            - tf.cast(
                tf.math.logical_xor(
                    (x[:, int(important_features[i1])] >= features_mean[i1]),
                    (x[:, int(important_features[i2])] >= features_mean[i2]),
                ),
                tf.float32,
            )
        )
        constraints.append(g)
        x_index += 1

    return constraints
