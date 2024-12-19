import numpy as np


def filter_correct_prediction(X_test, y_test, y_pred):
    index = y_test == y_pred
    return X_test[index], y_test[index], y_pred[index]


def filter_by_target_class(X_test, y_test, y_pred, target_class):
    index = y_test == target_class
    return X_test[index], y_test[index], y_pred[index]


def random_choice(X_test, y_test, y_pred, nb_sample):
    index = np.random.choice(X_test.shape[0], size=nb_sample, replace=False)
    return X_test[index], y_test[index], y_pred[index]
