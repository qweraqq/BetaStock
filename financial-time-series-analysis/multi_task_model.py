# -*- coding: UTF-8 -*-
from __future__ import division
import os
from keras.engine.training import Model
from keras.layers import Input,LSTM, Dense
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from helper import *
import numpy as np
import theano.tensor as T
from keras.utils.visualize_util import plot
__author__ = 'shenxiangxiang@gmail.com'
nb_hidden_units = 200
dropout = 0.2
l2_norm_alpha = 0.00001
nb_labels = 8


def custom_objective1(y_true, y_pred):
    """
    Custom objective function
    :param y_true: real value
    :param y_pred: predicted value
    :return: cost
    """
    # weight_matrix = ((y1 * y)<0)
    weight_matrix = 1 * ((y_true*y_pred) < 0)
    # T.abs_(y1-y)
    # (y1-y)**2
    # (weight_matrix)
    return T.mean(0.5*(1+weight_matrix)*(y_true-y_pred)**2)


def custom_objective2(y_true, y_pred):
    """
    Custom objective function
    :param y_true: real value
    :param y_pred: predicted value
    :return: cost
    """
    # weight_matrix = ((y1 * y)<0)
    weight_matrix = T.exp(T.abs_(y_true-y_pred))
    # T.abs_(y1-y)
    # (y1-y)**2
    # (weight_matrix)
    return T.mean(0.5*weight_matrix*(y_true-y_pred)**2)


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
        financial_time_series_input = Input(shape=(nb_time_step, dim_data), name='x1')
        lstm_layer_1 = LSTM(output_dim=nb_hidden_units, dropout_U=dropout, dropout_W=dropout,
                            W_regularizer=l2(l2_norm_alpha), b_regularizer=l2(l2_norm_alpha), activation='tanh',
                            return_sequences=True, name='lstm_layer1')
        lstm_layer_21 = LSTM(output_dim=nb_hidden_units, dropout_U=dropout, dropout_W=dropout,
                             W_regularizer=l2(l2_norm_alpha), b_regularizer=l2(l2_norm_alpha), activation='tanh',
                             return_sequences=True, name='lstm_layer2_loss1')
        lstm_layer_22 = LSTM(output_dim=nb_hidden_units, dropout_U=dropout, dropout_W=dropout,
                             W_regularizer=l2(l2_norm_alpha), b_regularizer=l2(l2_norm_alpha), activation='tanh',
                             return_sequences=True, name='lstm_layer2_loss2')
        lstm_layer_23 = LSTM(output_dim=nb_hidden_units, dropout_U=dropout, dropout_W=dropout,
                             W_regularizer=l2(l2_norm_alpha), b_regularizer=l2(l2_norm_alpha), activation='tanh',
                             return_sequences=True, name='lstm_layer2_loss3')

        lstm_layer_24 = LSTM(output_dim=nb_hidden_units, dropout_U=dropout, dropout_W=dropout,
                             W_regularizer=l2(l2_norm_alpha), b_regularizer=l2(l2_norm_alpha), activation='tanh',
                             return_sequences=True, name='lstm_layer2_loss4')

        lstm_layer_25 = LSTM(output_dim=nb_hidden_units, dropout_U=dropout, dropout_W=dropout,
                             W_regularizer=l2(l2_norm_alpha), b_regularizer=l2(l2_norm_alpha), activation='tanh',
                             return_sequences=True, name='lstm_layer2_loss5')
        h1 = lstm_layer_1(financial_time_series_input)
        h21 = lstm_layer_21(h1)
        h22 = lstm_layer_22(h1)
        h23 = lstm_layer_23(h1)
        h24 = lstm_layer_24(h1)
        h25 = lstm_layer_25(h1)
        time_series_predictions1 = TimeDistributed(Dense(1), name="p1")(h21)  # custom 1
        time_series_predictions2 = TimeDistributed(Dense(1), name="p2")(h22)  # custom 2
        time_series_predictions3 = TimeDistributed(Dense(1), name="p3")(h23)  # mse
        time_series_predictions4 = TimeDistributed(Dense(1, activation='sigmoid'), name="p4")(h24)  # logloss
        time_series_predictions5 = TimeDistributed(Dense(nb_labels, activation='softmax'), name="p5")(h25)  # cross
        self.model = Model(input=financial_time_series_input,
                           output=[time_series_predictions1, time_series_predictions2,
                                   time_series_predictions3, time_series_predictions4,
                                   time_series_predictions5],
                           name="multi-task deep rnn for financial time series forecasting")
        plot(self.model, to_file='model.png')

    def reset(self):
        for l in self.model.layers:
            if type(l) is LSTM:
                l.reset_status()

    def compile_model(self, lr=0.0001, arg_weight=1.):
        optimizer = Adam(lr=lr)
        loss = [custom_objective1, custom_objective2, 'mse', 'binary_crossentropy', 'categorical_crossentropy']
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit_model(self, X, y, y_label, epoch=300):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

        self.model.fit(X, [y]*3 + [y > 0] + [y_label], batch_size=self.batch_size, nb_epoch=epoch, validation_split=0.2,
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
        y_hat = self.model.predict(X, batch_size=1)[0]
        count_true = 0
        count_all = y.shape[1]
        for i in range(y.shape[1]):
            count_true = count_true + 1 if y[0,i,0]*y_hat[0,i,0]>0 else count_true
            print(y[0,i,0],y_hat[0,i,0])
        print(count_all,count_true)


if __name__ == '__main__':
    max_len = 200
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    training_set_path = os.path.join(this_file_path, "data",)
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

    training_set_path = os.path.join(this_file_path, "testdata",)
    file_list = os.listdir(training_set_path,)
    X_test = None
    y_test = None
    for idx, f in enumerate(file_list):
        file_path = os.path.join(training_set_path, f)
        X_tmp, y_tmp = read_single_financial_series_file(file_path,mode=1)
        X_tmp = sequence.pad_sequences(X_tmp, maxlen=max_len, dtype='float32')
        y_tmp = sequence.pad_sequences(y_tmp, maxlen=max_len, dtype='float32')
        X_test = X_tmp if X_test is None else np.vstack((X_test, X_tmp))
        y_test = y_tmp if y_test is None else np.vstack((y_test, y_tmp))

    y_label = np.zeros((y.shape[0], y.shape[1], nb_labels))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y_label[i, j, y_transform(y[i, j, 0])] = 1
    financial_time_series_model = FinancialTimeSeriesAnalysisModel(200, 5, batch_size=32,
                                                                   model_path="multask_ta.model.weights")
    financial_time_series_model.compile_model()
    financial_time_series_model.fit_model(X, y, y_label, epoch=500)
    financial_time_series_model.save()
    financial_time_series_model.model_eval(X_test, y_test)
