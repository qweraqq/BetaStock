# -*- coding: UTF-8 -*- 
from __future__ import division

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.preprocessing import sequence
from FundamentalAnalysis import *
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

model_architecture_file = "fundamental_analysis.model.json"
model_weights_file = "fundamental_analysis.model_weights.h5"

model = model_from_json(open(model_architecture_file).read())

model.load_weights(model_weights_file)
X_test, y_test = preprocessNews('201601news.txt', y_filename='sh.csv')
y_predict = model.predict(X_test,  batch_size=1)
for i in range(np.shape(y_test)[0]):  
    print y_test[i, 0], y_predict[i, 0]

a = model.get_weights()


def lstmOneStep(a, x_t, h_tm1=None, cell_tm1=None):
        
    Wxi = a[0]
    Whi = a[1]
    bi = a[2]

    Wxf =a[6]
    Whf = a[7]
    bf = a[8]
    
    Wxo = a[9]
    Who = a[10]
    bo = a[11]
    
    Wxc = a[3]
    Whc = a[4]
    bc = a[5]

    feature_dim = Wxo.shape[1] # output size
    if h_tm1 is None:
        h_tm1 = np.zeros((1, feature_dim))
    if cell_tm1 is None:
        cell_tm1 = np.zeros((1, feature_dim))
    input_gate = sigmoid(np.dot(x_t, Wxi) + np.dot(h_tm1, Whi) + bi)
    forget_gate = sigmoid(np.dot(x_t, Wxf) + np.dot(h_tm1, Whf) + bf)
    output_gate = sigmoid(np.dot(x_t, Wxo) + np.dot(h_tm1, Who) + bo)

    candidate_cell = np.tanh(np.dot(x_t, Wxc) + np.dot(h_tm1, Whc) + bc)

    cell = input_gate * candidate_cell + forget_gate * cell_tm1
    h_t = output_gate * np.tanh(cell)

    return h_t, cell

h_t, cell = lstmOneStep(a, X_test[0,189,:])
W = a[12]
b = a[13]
print np.dot(h_t, W)+b

