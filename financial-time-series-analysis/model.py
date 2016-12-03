# -*- coding: UTF-8 -*-
from __future__ import division
import os
from keras.engine.training import Model
from keras.layers import Input,LSTM, TimeDistributedDense
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

from helper import *
import numpy as np
__author__ = 'shenxiangxiang@gmail.com'

nb_hidden_units = 200
dropout = 0.1
l2_norm_alpha = 0.00001


class FinancialTimeSeriesAnalysisModel(object):
    model = None

    def __init__(self, nb_time_step, dim_data, batch_size=1, model_path=None):
        self.model_path = model_path
        self.model_path = model_path
        self.batch_size = batch_size
        self.size_of_input_data_dim = dim_data
        self.size_of_input_timesteps = nb_time_step
        self.build()
        self.weight_loaded = False
        if model_path is not None:
            self.load_weights()

    def build(self):
        dim_data = self.size_of_input_data_dim
        nb_time_step = self.size_of_input_timesteps
        financial_time_series_input = Input(shape=(nb_time_step, dim_data))

        lstm_layer_1 = LSTM(output_dim=nb_hidden_units, dropout_U=dropout, dropout_W=dropout, inner_activation='sigmoid',
                            W_regularizer=l2(l2_norm_alpha), b_regularizer=l2(l2_norm_alpha), activation='tanh',
                            return_sequences=True)
        lstm_layer_2 = LSTM(output_dim=nb_hidden_units, dropout_U=dropout, dropout_W=dropout, inner_activation='sigmoid',
                            W_regularizer=l2(l2_norm_alpha), b_regularizer=l2(l2_norm_alpha), activation='tanh',
                            return_sequences=True)

        h1 = lstm_layer_1(financial_time_series_input)
        h2 = lstm_layer_2(h1)
        time_series_predictions = TimeDistributedDense(1)(h2)
        self.model = Model(financial_time_series_input, time_series_predictions,
                           name="deep rnn for financial time series forecasting")

    def reset(self):
        for l in self.model.layers:
            if type(l) is LSTM:
                l.reset_status()

    def compile_model(self, lr=0.0001, arg_weight=1.):
        optimizer = Adam(lr=lr)
        loss = 'mse'
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit_model(self, X, y, X_val=None, y_val=None, epoch=3):
        early_stopping = EarlyStopping(monitor='val_loss',patience=3, verbose=0)
        if X_val is None:
            self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=epoch, validation_split=0.2,
                           shuffle=True, callbacks=[early_stopping])
        else:
            self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=epoch, validation_data=(X_val, y_val),
                           shuffle=True, callbacks=[early_stopping])

    def save(self):
        self.model.save_weights(self.model_path, overwrite=True)

    def load_weights(self):
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)
            self.weight_loaded = True

    def print_weights(self, weights=None, detail=False):
        weights = weights or self.model.get_weights()
        for w in weights:
            print("w%s: sum(w)=%s, ave(w)=%s" % (w.shape, np.sum(w), np.average(w)))
        if detail:
            for w in weights:
                print("%s: %s" % (w.shape, w))

    def model_eval(self, X, y):
        y_hat = self.model.predict(X, batch_size=1)
        count_true = 0
        count_all = y.shape[1]
        for i in range(y.shape[1]):
            count_true = count_true + 1 if y[0,i,0]*y_hat[0,i,0]>0 else count_true
            print(y[0,i,0], y_hat[0,i,0])
        print(count_all, count_true)


if __name__ == '__main__':
    max_len = 200
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    training_set_path = os.path.join(this_file_path,"data",)
    file_list = os.listdir(training_set_path,)
    X = None
    y = None
    for idx, f in enumerate(file_list):
        file_path = os.path.join(training_set_path, f)
        X_tmp, y_tmp = read_single_financial_series_file(file_path)
        X_tmp = sequence.pad_sequences(X_tmp, maxlen=max_len, dtype='float32')
        y_tmp = sequence.pad_sequences(y_tmp, maxlen=max_len, dtype='float32')
        X = X_tmp if X is None else np.vstack((X, X_tmp))
        y = y_tmp if y is None else np.vstack((y, y_tmp))
    nb_labels = 8
    y_label = np.zeros((y.shape[0], y.shape[1], nb_labels))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y_label[i, j, y_transform(y[i, j, 0])] = 1

    # training_set_path = os.path.join(this_file_path,"testdata",)
    # file_list = os.listdir(training_set_path,)
    # X_test = None
    # y_test = None
    # for idx, f in enumerate(file_list):
    #     file_path = os.path.join(training_set_path, f)
    #     X_tmp, y_tmp = read_single_financial_series_file(file_path,mode=1)
    #     X_tmp = sequence.pad_sequences(X_tmp, maxlen=max_len, dtype='float32')
    #     y_tmp = sequence.pad_sequences(y_tmp, maxlen=max_len, dtype='float32')
    #     X_test = X_tmp if X_test is None else np.vstack((X_test, X_tmp))
    #     y_test = y_tmp if y_test is None else np.vstack((y_test, y_tmp))

    # financial_time_series_model = FinancialTimeSeriesAnalysisModel(200,5,batch_size=32,model_path="ta.model.weights")
    # financial_time_series_model.compile_model()
    # financial_time_series_model.fit_model(X, y)
    # financial_time_series_model.save()
    # financial_time_series_model.model_eval(X_test, y_test)
