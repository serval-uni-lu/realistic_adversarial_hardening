import pickle
import glob


def load_from_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_from_dir(input_dir, handler=None):
    files_regex = input_dir + "/*.pickle"
    files = glob.glob(files_regex)
    obj_list = []

    for file_i, file in enumerate(files):
        with open(file, "rb") as f:
            obj = pickle.load(f)
            if handler is None:
                obj_list.append(obj)
            else:
                obj_list.append(handler(file_i, obj))
    return obj_list


def save_to_file(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
