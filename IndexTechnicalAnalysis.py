# -*- coding: UTF-8 -*-
from __future__ import division
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributedDense, Dropout
from keras.preprocessing import sequence
from keras.regularizers import l2, activity_l2

import numpy as np
import pandas as pd
import keras
import theano
import theano.tensor as T
from TechnicalAnalysis import readStockFile


def custom_objective(y_true, y_pred):
    """
    Custom objective function
    :param y_true: real value
    :param y_pred: predicted value
    :return: cost
    """

    y = T.as_tensor_variable(y_true)
    y1 = T.as_tensor_variable(y_pred)
    # weight_matrix = ((y1 * y)<0)
    weight_matrix = y < 0
    T.abs_(y1-y)
    # (y1-y)**2
    # (weight_matrix)
    return T.mean(0.5*(y1-y)**2)


if __name__ == '__main__':
    X_train ,y_train = readStockFile('train.csv', mode=1)
    data_dim = np.shape(X_train)[2]
    timesteps = np.shape(X_train)[1]
    nb_classes = 1
    nb_hidden = 50
    model = Sequential()
    model.add(LSTM(nb_hidden, input_dim=data_dim, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(nb_hidden, input_dim=nb_hidden, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(TimeDistributedDense(nb_classes, input_dim=nb_hidden,
                                   activation='linear', W_regularizer=l2(0.0001),
                                   b_regularizer=l2(0.001)))
    # model.add(Dropout(0.2))
    ##################
    model.compile(loss=custom_objective, optimizer='adagrad')
    
    X_val, y_val = readStockFile('val.csv', mode=1)
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=10, verbose=0, mode='auto')
    model.fit(X_train, y_train, batch_size=1, nb_epoch=100,
              validation_data=(X_val,y_val))#, callbacks=[earlyStopping])
    # model.load_weights('/home/shen/PycharmProjects/BetaStock/ta.model.weights.h5')
    X_test, y_test = readStockFile('test.csv', mode=1)

    y_predict = model.predict(X_test,  batch_size=1)
    for i in range(np.shape(y_test)[1]):
        if i>20:
            print y_test[0, i, 0], y_predict[0, i, 0]

    bin_result = (y_test*y_predict > 0)
    print np.bincount(bin_result[0, 20:, 0])
    print 'first is ', bin_result[0, 20, 0]
    json_string = model.to_json()
    file = open("sh.model.json", "w")
    file.write(json_string)
    file.close()
    model.save_weights('sh.model.weights.h5')
