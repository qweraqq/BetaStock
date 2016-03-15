# -*- coding: UTF-8 -*-
from __future__ import division
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributedDense, Dropout
from keras.preprocessing import sequence


import numpy as np
import pandas as pd
import keras
import os


def readStockFile(filename):
    df = pd.read_csv(filename, sep=',', header=0, usecols=[1, 2, 4, 7, 14])
    X = np.array(df)
    X[:, 3] = X[:, 3] / 100
    last_day_close_rev = 1/(1+X[:, 3])
    last_day_close_rev = last_day_close_rev.reshape((len(last_day_close_rev),1))
    df = pd.read_csv(filename, sep=',',header=0, usecols=[3])
    last_day_close = np.array(df) * last_day_close_rev

    X[:, 0:3] = ((X[:, 0:3] - last_day_close) / last_day_close)*100
    X[:, 4] = X[:, 4]
    nb_timesteps = np.shape(X)[0]
    X = X[np.newaxis, nb_timesteps-1:1:-1, :]
    df = pd.read_csv(filename, sep=',',header=0, usecols=[7])
    Y = np.array(df)
    y = []
    for val in Y:
        y.append(val)
    y = np.asarray(y)
    y = y[np.newaxis, nb_timesteps-2:0:-1, :]
    return X, y


def readAllData(dictname):
    X = None
    y = None
    file_list = os.listdir(dictname)
    for idx, f in enumerate(file_list):
        file_name = dictname+"/"+f
        (X_tmp, y_tmp) = readStockFile(file_name)
        X_tmp = sequence.pad_sequences(X_tmp, maxlen=280, dtype='float32')
        y_tmp = sequence.pad_sequences(y_tmp, maxlen=280, dtype='float32')
        if idx == 0:
            X = X_tmp
            y = y_tmp
            continue
        X = np.vstack((X, X_tmp))
        y = np.vstack((y, y_tmp))

    return X, y

if __name__ == '__main__':
    (X, y) = readAllData("./data")

    data_dim = np.shape(X)[2]
    timesteps = np.shape(X)[1]
    nb_classes = 1
    nb_hidden = 50
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    # model.add(BatchNormalization(mode=1,input_shape=(timesteps,data_dim)))
    model.add(LSTM(nb_hidden, input_dim=data_dim, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(Dropout(0.2))
    model.add(LSTM(nb_hidden, input_dim=nb_hidden, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributedDense(nb_classes, input_dim=nb_hidden, activation='linear'))

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    model.compile(loss='mean_squared_error', optimizer='Adagrad')  # rmsprop
    model.fit(X, y, batch_size=1, nb_epoch=2000, validation_split=0.4, callbacks=[earlyStopping], shuffle=True)

    (X_test, y_test) = readAllData("./test")
    score = model.evaluate(X_test, y_test, batch_size=1, verbose=1)
    print score
    y_predict = model.predict(X_test,  batch_size=1)

    for i in range(np.shape(y_test)[1]):
        if i > 200:
            print y_test[0, i, 0], y_predict[0, i, 0]

    json_string = model.to_json()
    file_to_save_model = open("sh.model.json", "w")
    file_to_save_model.write(json_string)
    file_to_save_model.close()
    model.save_weights('sh_model_weights.h5')
