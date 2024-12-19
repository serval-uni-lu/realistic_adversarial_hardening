import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
sys.path.append('..')
import time
import joblib
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fence.neris_attack_tf2 import Neris_attack
from pgd.pgd_attack_art import PgdRandomRestart
from training.helpers import  read_min_max
from tensorflow.random import set_seed
"""
set_seed(2)
from numpy.random import seed
seed(2)
"""
import random
np.random.seed(500)


import warnings
warnings.filterwarnings('ignore')


def attack(method, model_path, samples_path, labels_path, distance, iterations, mask_idx, eq_min_max, only_botnet=True): 
    samples = np.load(samples_path)
    labels = np.load(labels_path)
    if only_botnet:
        idx = np.where(labels==1)[0]
        labels = labels[idx]
        samples = samples[idx]
    model = load_model(model_path)

    if method == "pgd":
        labels = np.expand_dims(labels, axis=1)
        attack_generator = PgdRandomRestart(model, eps=distance, alpha = 1, num_iter = iterations, restarts = 5, scaler=scaler, mins=min_features, maxs=max_features, mask_idx=mask_idx, eq_min_max=eq_min_max)
        perturbSamples = attack_generator.run_attack(samples,labels)

    if method == "neris":
        perturbSamples = []
        attack_generator = Neris_attack(model_path = model_path, iterations=iterations, distance=distance, scaler=scaler, mins=min_features, maxs=max_features)
        for i in range(samples.shape[0]):
            if (i % 1000)==0:
                print("Attack ", i)
            sample = samples[i]
            sample = np.expand_dims(sample, axis=0)
            label = labels[i]
            adversary = attack_generator.run_attack(sample,label)
            perturbSamples.append(adversary)
        perturbSamples = np.squeeze(np.array(perturbSamples))

    probas = np.squeeze(model.predict(perturbSamples))
    predictions = np.squeeze((probas>= 0.5).astype(int))
    adv_idx = np.squeeze(np.argwhere(predictions == 0))
    success_rate = np.count_nonzero(predictions == 0)/predictions.shape[0]*100
    return perturbSamples, success_rate


if __name__ == "__main__":

    scaler = joblib.load('../data/neris/scaler.pkl')
    min_features, max_features = read_min_max('../data/neris/minimum.txt', '../data/neris/maximum.txt')
    mask_idx = np.load('../data/neris/mutable_idx.npy')
    eq_min_max = np.load('../data/neris/eq_min_max_idx.npy')
    start_time = datetime.now()
    perturbed_samples, success_rate_12 = attack('pgd',  '../out/datasets/neris/clean_10epochs/clean_trained_model.h5', '../data/neris/testing_samples.npy', '../data/neris/testing_labels.npy', distance=12, iterations=100, mask_idx=mask_idx, eq_min_max=eq_min_max)
    #np.save("perturbations_neris_all_nerisds_trainset.npy", perturbed_samples)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    print("Success rate", success_rate_12)