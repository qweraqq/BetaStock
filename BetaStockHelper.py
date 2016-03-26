# -*- coding: utf-8 -*-
from str2vec import *
from datetime import datetime, timedelta
from keras.preprocessing import sequence
import tushare as ts
import numpy as np
import pandas as pd
import os


def formatDateString(dt):
    """
    :param dt:
    :return:
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

def y_transform(y):
    """
    transform y(p_change) into labels
    :param y:
    :return:
    """

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
        X[:, 4] = X[:, 4] * 100
        X[:, 4] = X[:, 4] / 320000000000  # shanghai stock market total value
    X[:, 3] = X[:, 3]/100  # p_change
    tmp = X[1:, :]  # close, remove first day which do not have a previous close
    last_day_close = X[0:-1, 5]
    last_day_close = last_day_close.reshape((len(last_day_close), 1))
    tmp[:, 0:3] = (tmp[:, 0:3] - last_day_close) / last_day_close
    r_value = np.copy(tmp[:, 0:5])
    # Now r_value is as described
    # Do your own normalization
    r_value *= 10
    return r_value

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
        X = featureNormalization(X, mode=mode)
        r_X = X[np.newaxis, 0:-1, :]
        r_y = X[1:, 3]
        r_y = r_y.reshape((len(r_y),1))
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

if __name__ == '__main__':
    helper = BetaStockHelper()
    # news_rep1 = helper.str2Vec("平安称陆金所下半年启动上市 不受战新板不确定性影响")
    # news_rep2 = helper.str2Vec("财经观察：全球经济风险致美联储暂缓加息")
    # print news_rep1
    # print '--------------------------'
    # print np.sum(news_rep2**2)
    helper.readSingleFromFile('test.csv', mode=1)
    X, y = helper.readAllData('./data/')
    print y





