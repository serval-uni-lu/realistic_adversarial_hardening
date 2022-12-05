
import os
import sys
import time
import math

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision = 5)

        
class MDModel:
    def __init__(self, restore):
        network = Sequential()
        network.add(tf.keras.Input(shape=(756)))
        network.add(Dense(units = 256, activation = 'relu', input_dim = 756))
        network.add(Dense(units = 128, activation = 'relu'))
        network.add(Dense(units = 64, activation = 'relu'))
        network.add(Dense(units = 1)) 
        
        network.load_weights(restore)
        self.model = network

    def predict(self, data):
        return self.model(data)

def get_raw_delta(attack, delta, delta_sign, scaler, num_feature, attack_shape):

    raw_attack = scaler.inverse_transform(attack)
    adv = np.zeros(attack_shape)
    adv[0, num_feature] = delta * delta_sign
    adv += attack
    new_adv = scaler.inverse_transform(adv)
    new_delta = new_adv[0, num_feature] - raw_attack[0, num_feature]
    return new_delta

def get_scaled_delta(attack, delta, scaler, num_feature, attack_shape):

    raw_attack = scaler.inverse_transform(attack)
    adv = np.zeros(attack_shape)
    adv[0, num_feature] = delta
    adv +=raw_attack
    new_adv = scaler.transform(adv)
    new_delta = new_adv[0, num_feature] - attack[0, num_feature]
    return new_delta

def sigmoid(x):
    np.where(x < -100, 0, x) 
    return 1 / (1 + np.exp(-x))











