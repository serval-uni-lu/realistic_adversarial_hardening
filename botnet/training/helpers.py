import pickle
from csv import writer
import numpy as np
import joblib
from sklearn.metrics import f1_score,  confusion_matrix,  roc_auc_score, recall_score, precision_score, roc_curve, auc, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy


def create_DNN(units, input_dim_param, lr_param):
    network = Sequential()
    network.add(Input(shape=(756)))
    print(units)
    network.add(Dense(units = units[0], activation = 'relu'))
    network.add(Dropout(0.1))
    network.add(Dense(units = units[1], activation = 'relu'))
    network.add(Dropout(0.1))
    network.add(Dense(units = units[2], activation = 'relu'))
    network.add(Dense(units = 1))
    network.add(Activation('sigmoid'))
    sgd = Adam(lr = lr_param)
    network.compile( optimizer = sgd, loss=BinaryCrossentropy())
    return network


def save_metrics(y_test_array, predictions, metrics_path):
    f1 = f1_score(y_test_array, predictions)
    roc_auc = roc_auc_score(y_test_array, predictions)
    tn, fp, fn, tp = confusion_matrix(y_test_array, predictions).ravel()
    precision = precision_score(y_test_array, predictions)
    recall = recall_score(y_test_array, predictions)
    metrics_obj = {'f1':f1,
            'roc_auc': roc_auc,
            'fpr': fp/(fp+tn),
            'fnr': fn/(fn+tp)}
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics_obj, f)  

def save_adv_candidates(x_test_array, y_test_array, predictions, adv_candidates_path):
    idx = np.argwhere((predictions==y_test_array) & (y_test_array==1))
    idx = np.squeeze(idx)
    np.save(adv_candidates_path, x_test_array[idx])

def save_train_data(nn, x_test_array, y_test_array, metrics, adv_candidates_path):

    probas = nn.predict(x_test_array)
    predictions = np.squeeze((probas>= 0.5).astype(int))

    ## metrics
    save_metrics(y_test_array, predictions, metrics_path)

    ## adversarial candidates
    save_adv_candidates(y_test_array, predictions, adv_candidates_path)


def read_metrics(path):
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results


def add_data_to_df(data, df_path):
    with open(df_path, 'a',  newline="") as f_object:
        writer_object = writer(f_object)
        writer_object.writerows(data)
        f_object.close()


def get_adversarials(adv_candidates_path, model_path, distance, attack, adv_path):
    perturb_samples_pgd, pgd_success_rate = attack(attack, model_path, adv_candidates_path, distance, 100)
    np.save(adv_path, perturb_samples_pgd)



def read_min_max(min_file, max_file):

    with open(min_file, 'r') as f:
        mins = f.read()

    mins = mins.strip()
    mins = mins.replace(' ', '')
    mins_str_list = mins.split(',')
    min_features = [float(i) for i in mins_str_list]

    with open(max_file, 'r') as f:
        maxs = f.read()
    maxs = maxs.strip()
    maxs = maxs.replace(' ', '')
    maxs_str_list = maxs.split(',')
    max_features = [float(i) for i in maxs_str_list]

    return min_features, max_features




def get_train_data(config):
    x_train = np.load(config["x_train"])
    y_train = np.load(config["y_train"])
    x_test = np.load(config["x_test"])
    y_test = np.load(config["y_test"])
    return x_train, y_train, x_test, y_test

def get_model_data(config):
    LAYERS = config["LAYERS"]
    INPUT_DIM = config["INPUT_DIM"]
    LR = config["LR"]
    return LAYERS, INPUT_DIM, LR

def get_processing_data(config):
    scaler = joblib.load(config["scaler_path"])
    min_features, max_features = read_min_max(config["min_features"], config["max_features"])
    return scaler, min_features, max_features