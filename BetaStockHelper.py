# -*- coding: utf-8 -*-
from str2vec import *
from datetime import datetime, timedelta
from keras.preprocessing import sequence
import tushare as ts
import numpy as np
import pandas as pd
import os
import re


def to_categorical(y, nb_classes=8):
    '''
    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical crossentropy
    '''
    y = np.asarray(y, dtype='int32')
    Y = np.zeros((np.shape(y)[0], np.shape(y)[1], nb_classes))
    for i in range(np.shape(y)[0]):
        for j in range(np.shape(y)[1]):
            Y[i, j , y[i,j,0]] = 1
    return Y

def formatDateString(dt):
    """
    :param dt: type datetime
    :return: formatted datetime string
    """
    assert type(dt) is datetime, 'input type must be datetime type'
    rvalue = ""
    rvalue += str(dt.year)
    rvalue += "-"
    if dt.month < 10:
        rvalue += "0"
    rvalue += str(dt.month)
    rvalue += "-"
    if dt.day < 10:
        rvalue += "0"
    rvalue += str(dt.day)
    return rvalue


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
    if y <= -6:
        r = 0
    elif y > -6 and y <= -3:
        r = 1
    elif y > -3 and y <= -1:
        r = 2
    elif y > -1 and y <= 0:
        r = 3
    elif y > 0 and y <= 1:
        r = 4
    elif y > 1 and y <= 3:
         r = 5
    elif y > 3 and y <= 6:
        r = 6
    else:
        r = 7
    # if y <= 0:
    #     r = -1
    # else:
    #     r = 1
    # r = y
    # if y <= 0:
    #     r = 0
    # else:
    #     r = 1
    return r

def featureNormalization(X, mode=0):
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
    r_y = np.copy(X[2:, 3])*100
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

class BetaStockHelper(object):
    """
    Help do some pre-process work
    """

    def __init__(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        allowed_kwargs = {'name',
                          'word_emb_file',
                          'word_emb_vocab',
                          'rae_param_dict',
                          'todo_args'}

        for kwarg in kwargs:
            assert kwarg in allowed_kwargs, 'keyword argument not understood' + kwarg

        if 'name' in kwargs:
            self.name = kwargs['name']

        # load word embeddings
        if 'word_emb_file' in kwargs:
            self.word_emb_file = kwargs['word_emb_file']
        else:
            self.word_emb_file = './data_preserved/word.vectors.txt'

        if 'word_emb_vocab' in kwargs:
            self.word_emb_vocab = kwargs['word_emb_vocab']
        else:
            self.word_emb_vocab = './data_preserved/word.vocab.txt'
        # word embeddings, vocab
        self.W_norm, self.vocab, _ = \
            loadWordEmbeddings(self.word_emb_vocab,
                               self.word_emb_file)

        # load RAE parameters
        if 'rae_param_dict' in kwargs:
            self.rae_param_dict = kwargs['rae_param_dict']
        else:
            self.rae_param_dict = './data_preserved/'

        self.rae_W1, self.rae_W2, self.rae_b = \
            loadRaeParameters(self.rae_param_dict+'W1.txt',
                              self.rae_param_dict+'W2.txt',
                              self.rae_param_dict+'b.txt')

    def str2Vec(self, news_headline):
        """
        :param news_headline: type string, like 'lianghui shunli zhaokai'
        :return: distributed rep for the news headline
        """
        news_tokenized = tokenizeSentence(news_headline,
                                          self.W_norm, self.vocab)
        news_rep = strToVector(news_tokenized,
                               self.rae_W1, self.rae_W2, self.rae_b)

        return news_rep

    def getNextDayReturn(self, dt=None, max_day_try=10):
        """
        given a date, return the return next day
        :param dt: type datetime
        :param max_day_try: type int, to skip stock breaks, default 10
        :return: None if invalid, return_next_day otherwise
        """
        if type(dt) is not datetime:
            return None

        assert max_day_try >= 1, 'at least one day'

        dt1 = dt + timedelta(days=1)
        dt2 = dt + timedelta(days=max_day_try)
        stock_data = ts.get_hist_data('sh', start=formatDateString(dt1),
                                      end=formatDateString(dt2), retry_count=10)
        if stock_data.empty:
            return None

        return stock_data.as_matrix(['p_change'])[-1]
        # since the return value is reversed ordered

    def getPreviousStockData(self, dt=None, td=200, max_len=200):
        """
        :param dt: type-datetime ,datetime of the news
        :param td: time delta, number days to backtrack, default 200
        :param max_len: padding parameters, default 200
        :return:
        """
        if type(dt) is not datetime:
            dt = datetime.today()

        dt2 = dt - timedelta(days=td)

        stock_data = ts.get_hist_data('sh', start=self.formatDateString(dt2),
                                      end=self.formatDateString(dt), retry_count=10)

        stock_data = stock_data.as_matrix(['open', 'high', 'low', 'p_change', 'volume', 'close'])
        stock_data = stock_data[stock_data.shape[0]::-1, :]  # reserve
        stock_data = featureNormalization(stock_data, mode=1)
        return stock_data

    def _getStockDate(self, dt):
        """

        :param dt:
        :return:
        """
        if type(dt) is not datetime:
            return None
        if dt.hour < 15:  # 当天交易日
            dt = datetime(dt.year, dt.month, dt.day, 0, 0)
        else:  # 下一交易日
            dt = datetime(dt.year, dt.month, dt.day, 0, 0) + timedelta(days=1)
        return dt

    def readSingleFromFile(self, filename, mode=0):
        """
        :param filename: saved stock file, like '600000.csv'
        :param mode: mode 0 - regular stocks
                     mode 1 - index
        :return: tuple (X ,y)
                 X = (1,nb_samples,nb_features) 3d tensor
                 y = (1,nb_samples,1) 3d tensor
                 here, nb_samples a.k.a timesteps
        """
        if mode == 0:
            df = pd.read_csv(filename, sep=',', header=0, usecols=[1, 2, 4, 7, 14, 3])
            X = df.as_matrix(['open', 'high', 'low', 'p_change', 'turnover', 'close'])

        if mode == 1:
            df = pd.read_csv(filename, sep=',', header=0, usecols=[1, 2, 4, 5, 7, 3])
            X = df.as_matrix(['open', 'high', 'low', 'p_change', 'volume', 'close'])
        X = X[::-1, :]
        X,r_y = featureNormalization(X, mode=mode)
        r_X = X[np.newaxis, 0:-1, :]

        ###### y value transformation

        r_y  = y_t(r_y)
        r_y = r_y.reshape((len(r_y), 1))
        r_y = r_y[np.newaxis, :, :]
        return r_X, r_y

    def readAllData(self, dictname, maxlen=300):
        """
        read all files in dictname
        :param dictname: dictionary name
        :param maxlen:  type int, padding parameters
        :return: tuple (X, y)
                 X = (nb_samples, timesteps, nb_features)
                 y = (nb_samples, timesteps, nb_labels)
                 timesteps = maxlen
        """
        X = None
        y = None
        file_list = os.listdir(dictname)
        for idx, f in enumerate(file_list):
            file_name = dictname+f
            X_tmp, y_tmp = self.readSingleFromFile(file_name)
            X_tmp = sequence.pad_sequences(X_tmp, maxlen=maxlen, dtype='float32')
            y_tmp = sequence.pad_sequences(y_tmp, maxlen=maxlen, dtype='float32')
            if idx == 0:
                X = X_tmp
                y = y_tmp
                continue
            X = np.vstack((X, X_tmp))
            y = np.vstack((y, y_tmp))
        return X, y

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
                                        vocab_file='./data_preserved/word.vocab.txt',
                                        vectors_file='./data_preserved/word.vectors.txt',
                                        W1_file='./data_preserved/W1.txt',
                                        W2_file='./data_preserved/W2.txt',
                                        b_file='./data_preserved/b.txt')
        if idx == 0:
            X = X_tmp
            y = y_tmp
            continue

        X = np.vstack((X, X_tmp))
        y = np.vstack((y, y_tmp))

    return X, y


if __name__ == '__main__':
    helper = BetaStockHelper()
    # news_rep1 = helper.str2Vec("平安称陆金所下半年启动上市 不受战新板不确定性影响")
    # news_rep2 = helper.str2Vec("财经观察：全球经济风险致美联储暂缓加息")
    # print news_rep1
    # print '--------------------------'
    # print np.sum(news_rep2**2)
    X, y = helper.readAllData('I:\\BetaStock\\data\\')
    print y





