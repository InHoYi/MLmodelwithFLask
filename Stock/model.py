import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
# from keras.callbacks import EarlyStopping

number_of_features = 1
window_size = 4

seed = 42
np.random.seed(seed)       
random.seed(seed)          
tf.random.set_seed(seed)

def get_model(dataset):
    model = keras.Sequential()

    model.add(keras.layers.LSTM(64, input_shape = (window_size, number_of_features), activation = 'relu', return_sequences = True)) 
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.LSTM(32, return_sequences = True)) 
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.LSTM(16, return_sequences = False)) 
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(16))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(8))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(1))

    return model

def train(model, epochs, train_X, train_Y, valid_X, valid_Y):

    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
    # earlyStopping = EarlyStopping( monitor = 'val_loss', patience = 30, verbose = 1, restore_best_weights = True)
    # history = model.fit(train_X, train_Y, epochs = epochs, batch_size = 32, validation_data = (valid_X, valid_Y), callbacks = [earlyStopping])
    history = model.fit(train_X, train_Y, epochs = epochs, batch_size = 16, validation_data = (valid_X, valid_Y))

    return history

def print_summary(model):
    result = model.summary()
    print(result)