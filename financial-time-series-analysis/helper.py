# -*- coding: UTF-8 -*-
from __future__ import division
__author__ = 'shenxiangxiang@gmail.com'

import pandas as pd
import numpy as np

def time_series_normalization(X, mode=0):
    """
    Do your own normalization
    :param X: (nb_examples, nb_features) matrix
              here, nb_samples a.k.a timesteps
              X[: ,0] = (open-last_day_close)/last_day_close
              X[: ,1] = (high-last_day_close)/last_day_close
              X[: ,2] = (low-last_day_close)/last_day_close
              X[: ,3] = (close-last_day_close)/last_day_close
              X[: ,4] = turnover rate or volume if index(mode1)
              X[: ,5] = close (not included in return value)
    :param mode: is it index or stock, mode0-stock, mode1-index
    :return: normalized X
    """
    if mode == 1:
        X[:, 4] = X[:, 4] * 1
        X[:, 4] = X[:, 4] / 250000000  # shanghai stock market total value
    else:
        X[:, 4] /= 100
    X[:, 3] = X[:, 3]/100  # p_change
    r_y = np.copy(X[2:, 3])
    tmp = X[1:, :]  # close, remove first day which do not have a previous close
    last_day_close = X[0:-1, 5]
    last_day_close = last_day_close.reshape((len(last_day_close), 1))
    tmp[:, 0:3] = (tmp[:, 0:3] - last_day_close) / last_day_close
    r_value = np.copy(tmp[:, 0:5])
    # Now r_value is as described
    # Do your own normalization
    # r_value *= 10
    r_value = np.sign(r_value)*np.log(1+np.abs(r_value*100))
    return r_value, r_y


## very important
def y_t(y):
    for idx,_ in enumerate(y):
        y[idx] = y_transform(y[idx])
    return y

def y_transform(y):
    """
    transform y(p_change) into labels
    :param y:
    :return:
    """
    y = y*100
    if y <= -3:
        r = 0
    elif y > -3 and y <= -1:
        r = 1
    elif y > -1 and y <= -0.5:
        r = 2
    elif y > -0.5 and y <= 0:
        r = 3
    elif y > 0 and y <= 0.5:
        r = 4
    elif y > 0.5 and y <= 1:
         r = 5
    elif y > 1 and y <= 3:
        r = 6
    else:
        r = 7
    return r


def read_single_financial_series_file(filename, mode=0):
    """
    :param filename: saved stock file, like '600000.csv'
    :param mode: mode 0 - regular stocks
                 mode 1 - index
    :return: tuple (X ,y)
             X = (1,nb_samples,nb_features) 3d tensor
             y = (1,nb_samples,1) 3d tensor
             here, nb_samples a.k.a timesteps
    """
    X = None
    if mode == 0:
        df = pd.read_csv(filename, sep=',', header=0, usecols=[1, 2, 4, 7, 14, 3])
        X = df.as_matrix(['open', 'high', 'low', 'p_change', 'turnover', 'close'])

    if mode == 1:
        df = pd.read_csv(filename, sep=',', header=0, usecols=[1, 2, 4, 5, 7, 3])
        X = df.as_matrix(['open', 'high', 'low', 'p_change', 'volume', 'close'])

    X = X[::-1, :]
    X,r_y = time_series_normalization(X, mode=mode)
    r_X = X[np.newaxis, 0:-1, :]
    ###### y value transformation
    # r_y  = y_t(r_y)
    r_y = r_y.reshape((len(r_y), 1))
    r_y = r_y[np.newaxis, :, :]
    return r_X, r_y