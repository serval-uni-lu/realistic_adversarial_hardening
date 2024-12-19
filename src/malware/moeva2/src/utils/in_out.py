import sys

import joblib
import numpy as np
import pickle
import json
import glob
from tqdm import tqdm
from joblib import Parallel, delayed

# Using pickle


def pickle_from_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_from_dir(input_dir, handler=None, n_jobs=1):
    files_regex = input_dir + "/*.pickle"
    files = glob.glob(files_regex)

    if handler == None:
        handler = lambda i, x: id_function(x)

    def load_handle(file_i, file):
        with open(file, "rb") as f:
            obj = pickle.load(f)
            return handler(file_i, obj)

    return Parallel(n_jobs=n_jobs)(
        delayed(load_handle)(file_i, file)
        for file_i, file in tqdm(enumerate(files), total=len(files))
    )


def pickle_to_file(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# Using numpy array


def load_from_file(path):
    return np.load(path)


def id_function(obj):
    return obj


def load_from_dir(input_dir, handler=None, n_jobs=-1):

    files_regex = input_dir + "/*.npy"
    files = glob.glob(files_regex)
    obj_list = []

    for file_i, file in enumerate(files):
        obj = np.load(file)
        if handler is None:
            obj_list.append(obj)
        else:
            obj_list.append(handler(file_i, obj))
    return obj_list


def save_to_file(obj, path):
    with open(path, "wb") as f:
        np.save(f, obj)


# Using json files


def json_from_file(path):
    with open(path, "r") as f:
        return json.load(f)


def json_to_file(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def json_from_dir(input_dir, handler=None):
    files_regex = input_dir + "/*.json"
    files = glob.glob(files_regex)
    obj_list = []

    for file_i, file in enumerate(files):
        with open(file, "rb") as f:
            obj = json.load(f)
            if handler is None:
                obj_list.append(obj)
            else:
                obj_list.append(handler(file_i, obj))
    return obj_list


def get_parameters():

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "../config/default.json"

    return json_from_file(config_path)


def load_model(path: str):

    model = None
    if path.endswith(".joblib"):
        import joblib

        model = joblib.load(path)

    if path.endswith(".model"):
        from tensorflow.keras.models import load_model

        model = load_model(path)

    if model is None:
        raise NotImplementedError

    return model
