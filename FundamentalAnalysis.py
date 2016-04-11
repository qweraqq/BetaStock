# -*- coding: UTF-8 -*-
from __future__ import division
from seya.layers.recurrent import Bidirectional
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.preprocessing import sequence
from keras.regularizers import l2
import keras
# import tushare as ts
from BetaStockHelper import *
import logging
import theano.tensor as T
# from keras.layers.recurrent import Recurrent

# logging.basicConfig(level=logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Fundamental Analysis.py')


def custom_objective(y_true, y_pred):
    """
    Custom objective function
    :param y_true: real value
    :param y_pred: predicted value
    :return: cost
    """
    # weight_matrix = ((y1 * y)<0)
    # weight_matrix = T.exp(T.abs_(y_true-y_pred)/50)
    # T.abs_(y1-y)
    # (y1-y)**2
    weight_matrix = T.log10(1+T.abs_(y_true))
    return T.mean(0.5*weight_matrix*(y_true-y_pred)**2)


if __name__ == '__main__':
    # X, y = preprocessNews('201511news.txt', y_filename='sh.csv')
    X, y = readAllData('.\\newsdata\\', y_filename='.\\test\\sh.csv')
    X_test, y_test = preprocessNews('.\\2016news.txt', y_filename='sh.csv', year='2016年',
                                    vocab_file='.\\data_preserved\\word.vocab.txt',
                                    vectors_file='.\\data_preserved\\word.vectors.txt',
                                    W1_file='.\\data_preserved\\W1.txt',
                                    W2_file='.\\data_preserved\\W2.txt',
                                    b_file='.\\data_preserved\\b.txt')

    data_dim = np.shape(X)[2]
    timesteps = np.shape(X)[1]
    nb_classes = 1
    nb_hidden = 200
    model = Sequential()
    lstm1 = LSTM(output_dim=nb_hidden, input_dim=data_dim,
                 dropout_U=0.6, dropout_W=0.6, W_regularizer=l2(0.01), b_regularizer=l2(0.01))
    lstm2 = LSTM(output_dim=nb_hidden, input_dim=data_dim,
                 dropout_U=0.6, dropout_W=0.6, W_regularizer=l2(0.01), b_regularizer=l2(0.01))
    blstm = Bidirectional(forward=lstm1, backward=lstm2)
    blstm.set_input_shape((np.shape(X)[0], timesteps, data_dim))
    model.add(blstm)
    model.add(Dropout(0.6))
    model.add(Dense(1, W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
    # model.add(Dropout(0.5))
    # model.add(Dense(nb_classes, input_dim=2*nb_hidden, activation='linear'))
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=0, verbose=0, mode='auto')
    model.compile(loss=custom_objective, optimizer='rmsprop')
    model.fit(X, y, batch_size=10, nb_epoch=10, validation_data=(X_test, y_test),
              shuffle=True, callbacks=[earlyStopping])

    json_string = model.to_json()

    file_to_save_model = open("fa.model.json", "w")
    file_to_save_model.write(json_string)
    file_to_save_model.close()
    model.save_weights('fa.model_weights.h5')

    y_predict = model.predict(X_test,  batch_size=1)

    for i in range(np.shape(y_test)[0]):  
        print y_test[i, 0], y_predict[i, 0]

    
   

