# -*- coding: UTF-8 -*-
from __future__ import division
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributedDense, Dropout
from keras.preprocessing import sequence
from keras.regularizers import l2
import numpy as np
import pandas as pd
import keras
import os
import theano.tensor as T
from theano import function, printing
from BetaStockHelper import *
from theano import config

# config.exception_verbosity='high'
# config.compute_test_value = 'warn'
def custom_objective(y_true, y_pred):
    """
    Custom objective function
    :param y_true: real value
    :param y_pred: predicted value
    :return: cost
    """
    # weight_matrix = ((y1 * y)<0)
    weight_matrix = 0.5*((y_true*y_pred) < 0)
    # T.abs_(y1-y)
    # (y1-y)**2
    # (weight_matrix)
    return T.mean(0.5*(1+weight_matrix)*(y_true-y_pred)**2)

def custom_objective1(y_true, y_pred):
    """
    Custom objective function
    :param y_true: real value
    :param y_pred: predicted value
    :return: cost
    """
    # weight_matrix = ((y1 * y)<0)
    weight_matrix = T.exp(T.abs_(y_true-y_pred)/10)
    # T.abs_(y1-y)
    # (y1-y)**2
    # (weight_matrix)
    return T.mean(0.5*(weight_matrix)*(y_true-y_pred)**2)

if __name__ == '__main__':
    nb_classes = 8  # Output size
    nb_hidden = 100  # num hidden units
    helper = BetaStockHelper()
    X_train, y_train = helper.readAllData("./data/")
    y_train = to_categorical(y_train)
    print y_train
    #y_train= to_categorical(y_train, nb_classes=8)
    data_dim = np.shape(X_train)[2]
    timesteps = np.shape(X_train)[1]

    model = Sequential()
    model.add(LSTM(nb_hidden, input_dim=data_dim, return_sequences=True,
                   W_regularizer=l2(0.001), b_regularizer=l2(0.01)))
    model.add(Dropout(0.7))
    model.add(LSTM(nb_hidden, input_dim=nb_hidden, return_sequences=True,
                   W_regularizer=l2(0.001), b_regularizer=l2(0.01)))
    model.add(Dropout(0.7))
    model.add(TimeDistributedDense(nb_classes, input_dim=nb_hidden,
                                   activation='softmax', W_regularizer=l2(0.001),
                                   b_regularizer=l2(0.01)))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=30, verbose=0, mode='auto')

    model.fit(X_train, y_train, batch_size=100, nb_epoch=300,
              validation_split=0.3, callbacks=[earlyStopping],
              shuffle=True, show_accuracy=False)
    json_string = model.to_json()
    file_to_save_model = open("ta_mulloss_trans.model.json", "w")
    file_to_save_model.write(json_string)
    file_to_save_model.close()
    model.save_weights('ta_mulloss_trans.model.weights.h5')

    X_test, y_test = helper.readSingleFromFile("./test/sh.csv", mode=1)
    y_test = to_categorical(y_test, nb_classes=8)
    score = model.evaluate(X_test, y_test, batch_size=1, verbose=1)
    y_predict = model.predict(X_test,  batch_size=1)
    np.savetxt('y_test_mulloss_trans.txt', y_test.reshape((y_test.shape[1], y_test.shape[2])))
    np.savetxt('y_predict_mulloss_trans.txt', y_predict.reshape((y_test.shape[1], y_test.shape[2])))
