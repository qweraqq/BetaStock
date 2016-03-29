# -*- coding: UTF-8 -*-
from __future__ import division
from seya.layers.recurrent import Bidirectional
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.preprocessing import sequence
from keras.regularizers import l2
from datetime import datetime, date, time, timedelta
import numpy as np
import pandas as pd
import re
import keras
import os
# import tushare as ts
from str2vec import loadRaeParameters, strToVector, loadWordEmbeddings, tokenize_sentence
import logging
import types
import theano
import theano.tensor as T
from keras.layers.recurrent import Recurrent

# logging.basicConfig(level=logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Fundamental Analysis.py')


def preprocessNews(filename, y_filename=None, vocab_file='jieba_ths_vocab_big.txt',
                   vectors_file='jieba_ths_vectors_big.txt',
                   W1_file='W1.txt', W2_file='W2.txt', b_file='b.txt',
                   maxlen=200, year='2015年'):
    """
    Convert a file of news headlines to (X, y) tuple
    (X, y) is feed into keras

    #Input: filename
            vocab_file
    """
    X = None
    y = None
    y_dict = None
    
    # load word embeddings
    W_norm, vocab, ivocab = loadWordEmbeddings(vocab_file, vectors_file)

    # load RAE parameters
    W1, W2, b = loadRaeParameters(W1_file, W2_file, b_file)

    # load y
    if y_filename:
        df = pd.read_csv(y_filename, sep=',',header=0, usecols=[0,7])
        y_dict = {datetime.strptime(date, "%Y-%m-%d"): (df['p_change'][idx])
                  for idx, date in enumerate(df['date'])}
   
    base_time = datetime(1991, 12, 20, 0, 0)
    one_day_output = []
    with open(filename, 'r') as f:    
        for line in f.readlines():
            line = line.strip()
            if len(line) < 1: 
                continue  # skip if empty line
            if re.match(r'^\d+月\d+日 \d+:\d+$', line):
                line_time = datetime.strptime(year+line, '%Y年%m月%d日 %H:%M')
                line_time = getStockDate(line_time)
                
                if (line_time-base_time).days >= 1 and len(one_day_output)==0:
                    base_time = line_time
                    one_day_output.append(strToVector(tokenized_str, W1, W2, b))
                elif (line_time-base_time).days >= 1 and len(one_day_output)>0:                   
                    y_tmp = [[getStockReturn(base_time, y_dict)]]
                    y_tmp = sequence.pad_sequences(y_tmp, maxlen=1, dtype='float32')
                    
                    if y is None:
                        y = y_tmp
                    else:                
                        y = np.vstack((y, y_tmp))
                     
                    X_tmp = mergeTokenizedStr(one_day_output, maxlen)
                    if X is None:
                        X = X_tmp
                    else:                     
                        X = np.vstack((X, X_tmp))
               
                    base_time = line_time
                    one_day_output = []
                    one_day_output.append(strToVector(tokenized_str, W1, W2, b))
                else:
                    one_day_output.append(strToVector(tokenized_str, W1, W2, b))
     
            else: # if not match datetime line
                tokenized_str = tokenize_sentence(line, W_norm, vocab)

    if X is not None:
        y_tmp = [[getStockReturn(base_time, y_dict)]]
        y_tmp = sequence.pad_sequences(y_tmp, maxlen=1, dtype='float32')
        y = np.vstack((y, y_tmp))
        X_tmp = mergeTokenizedStr(one_day_output, maxlen)
        X = np.vstack((X, X_tmp))
        
    return X, y


def mergeTokenizedStr(one_day_output, maxlen=200):
    X_tmp = np.vstack(one_day_output)
    X_tmp = X_tmp[np.newaxis, :, :]
    X_tmp = sequence.pad_sequences(X_tmp, maxlen=maxlen, dtype='float32')   
    return X_tmp    

        
def getStockDate(dt):
    """
    return the actual return date of date dt
    #Input: date time type dt
    """
    
    if dt.hour < 15: # 当天交易日
        dt = datetime(dt.year, dt.month, dt.day, 0, 0)
    else: # 下一交易日
        dt = datetime(dt.year, dt.month, dt.day, 0, 0) + timedelta(days=1)
    return dt


def getStockReturn(dt, history_dict=None):
    if history_dict is None:  # TODO use tushare
        return 0

    tmp_time = dt
    while (not history_dict.has_key(tmp_time)):
        tmp_time = tmp_time + timedelta(days=1)
    return history_dict[tmp_time]


def readAllData(dictname, y_filename=None):
    X = None
    y = None
    file_list = os.listdir(dictname,)
    for idx, f in enumerate(file_list):
        file_name = dictname+f
        logger.info(file_name)
        (X_tmp, y_tmp) = preprocessNews(file_name, y_filename=y_filename,
                                        vocab_file='.\\data_preserved\\word.vocab.txt',
                                        vectors_file='.\\data_preserved\\word.vectors.txt',
                                        W1_file='.\\data_preserved\\W1.txt',
                                        W2_file='.\\data_preserved\\W2.txt',
                                        b_file='.\\data_preserved\\b.txt')
        if idx == 0:
            X = X_tmp
            y = y_tmp
            continue

        X = np.vstack((X, X_tmp))
        y = np.vstack((y, y_tmp))

    return X, y


if __name__ == '__main__':
    # X, y = preprocessNews('201511news.txt', y_filename='sh.csv')
    X, y = readAllData('.\\data\\', y_filename='I:\\BetaStock\\test\\sh.csv')

    data_dim = np.shape(X)[2]
    timesteps = np.shape(X)[1]
    nb_classes = 1
    nb_hidden = 200
    model = Sequential()
    lstm1 = LSTM(output_dim=nb_hidden, input_dim=data_dim)
    lstm2 = LSTM(output_dim=nb_hidden, input_dim=data_dim)
    blstm = Bidirectional(forward=lstm1, backward=lstm2)
    blstm.set_input_shape((np.shape(X)[0],timesteps, data_dim))
    model.add(blstm)

    model.add(Dense(1))
    # model.add(Dropout(0.5))
    # model.add(Dense(nb_classes, input_dim=2*nb_hidden, activation='linear'))
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=2, verbose=0, mode='auto')
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.fit(X, y, batch_size=10, nb_epoch=3, validation_split=0.3,
              callbacks=[earlyStopping], shuffle=True)

    X_test, y_test = preprocessNews('.\\test\\201601news.txt', y_filename='sh.csv', year='2016年')
    y_predict = model.predict(X_test,  batch_size=1)

    for i in range(np.shape(y_test)[0]):  
        print y_test[i, 0], y_predict[i, 0]
    json_string = model.to_json()
    
    file_to_save_model = open("newfa.model.json", "w")
    file_to_save_model.write(json_string)
    file_to_save_model.close()
    model.save_weights('newfa.model_weights.h5')
    
   

