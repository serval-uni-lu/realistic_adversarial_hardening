import sys
import os
#os.environ['PYTHONHASHSEED']=str(2)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random as rn
import json
import numpy as np
import tensorflow as tf
import pickle

from datagen import generate_adversarial_batch_pgd, generate_adversarial_batch_fence
from helpers import create_DNN, save_metrics, save_adv_candidates, read_min_max
from helpers import get_train_data, get_model_data, get_processing_data
#rn.seed(2)
#tf.random.set_seed(2)
#np.random.seed(2)

import warnings
warnings.filterwarnings('ignore')



def train(config, method='clean', callback=None, distance=12,  save_data=True):

    x_train, y_train, x_test, y_test = get_train_data(config)
    LAYERS, INPUT_DIM, LR = get_model_data(config)
    scaler, min_features, max_features = get_processing_data(config)
    iterations = config["iterations"]
    epochs = config["epochs"]
    ##Only for FENCE attack
    intermediate_model_path = config["intermediate_model_path"]

    for lrate in LR:
        nn =  create_DNN(units = LAYERS, input_dim_param = INPUT_DIM, lr_param = lrate)

        if method == "clean":
            history_obj = nn.fit(x_train, y_train, verbose=1, epochs=epochs, batch_size=64,  shuffle=True)

        if method =="pgd":
            dataGen = generate_adversarial_batch_pgd(nn, 64, x_train, y_train, distance, iterations, scaler=scaler, mins=min_features, maxs=max_features)
            history_obj = nn.fit(dataGen, steps_per_epoch=len(x_train) // 64, verbose = 1, epochs = epochs, callbacks=callback)

        if method == "fence":
            dataGen = generate_adversarial_batch_fence(nn, 64, x_train, y_train, distance, iterations, scaler, min_features, max_features, intermediate_model_path)
            history_obj = nn.fit(dataGen, steps_per_epoch=len(x_train) // 64, verbose = 1, epochs = epochs, callbacks=callback,)

        if save_data==True:
            ## model
            model_path = config["path_to_save"] + f"/{attack}_model.h5"
            nn.save(model_path)

            ## history
            history_path = config["path_to_save"] + f"/{attack}_model_history.npy"
            with open(history_path , 'wb') as f:
                pickle.dump(history_obj.history, f) 

            probas = np.squeeze(nn.predict(x_test))
            predictions = np.squeeze((probas>= 0.5).astype(int))
            ## metrics
            metrics_path = config["path_to_save"] + f"/{attack}_model_metrics.pickle"
            save_metrics(y_test, predictions, metrics_path)

            ## adversarial candidates
            adv_candidates_path = config["path_to_save"] + "initial_states.npy"
            save_adv_candidates(x_test, y_test, predictions, adv_candidates_path)



def train_save_epochs(config, attack):
    file = open(config_file)
    config = json.load(file)
    distances = config["distances"]
    print(distances)
    for i, dis in enumerate(distances):
        checkpoint_path =  config["path_to_save"] + f"/distance_{distances[i]}"
        checkpoint_path = checkpoint_path + "/model-{epoch:04d}.h5"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_freq="epoch", save_weights_only=False,  save_best_only=False, verbose=1)
        train(config, method=attack,  callback=cp_callback, distance=int(dis), save_data=True)



if __name__ == "__main__":
    
    config_file = "config/neris.json" 
    attack = "fence" 

    train_save_epochs(config_file, attack)

    


 





