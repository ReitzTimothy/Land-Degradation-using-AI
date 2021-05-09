import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import numpy as np
import sklearn as skl

class Critic:
    model = None

    def __init__(self, input_shape):
        inp = Input(shape=input_shape)
        x = Dense(1024, activation = 'selu')(inp)
        x = Dense(1024, activation = 'selu')(x)
        x = Dense(1024, activation = 'selu')(x)
        x = Dense(1024, activation = 'selu')(x)
        x = Dense(1, activation = 'sigmoid')(x)
        self.model = Model(inp, x)
    
    def train_critic_on_batch(self, data_generated, data_real):
        
        out = np.zeros((data_generated.shape[0]*2))
        for i in range(data_generated.shape[0]):
            out[i] = 1
            out[i+data_generated.shape[0]] = 0

        inp = np.concatenate((data_generated, data_real), axis = 0)
        loss = self.model.train_on_batch(inp, out)
        return loss