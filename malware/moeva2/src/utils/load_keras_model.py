from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
class MDModel:
    def __init__(self, restore):
        
        network = Sequential()
        network.add(Dense(units = 256, activation = 'relu', input_dim = 756))
        network.add(Dense(units = 128, activation = 'relu'))
        network.add(Dense(units = 64, activation = 'relu'))
        network.add(Dense(units = 1))   
        
        network.load_weights(restore)

        self.model = network

    def predict(self, data):
        return self.model(data)


