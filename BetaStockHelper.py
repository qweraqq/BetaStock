# -*- coding: utf-8 -*-
from str2vec import *
from datetime import datetime, timedelta
import tushare as ts
import numpy as np

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
              X[: ,5] = close
    :param mode: is it index or stock, mode0-stock, mode1-index
    :return: normalized X
    """

    if mode == 1:
        X[:, 4] = X[:, 4] * 100
        X[:, 4] = X[:, 4] / 320000000000  # shanghai stock market total value
    X[:, 3] = X[:, 3]/100
    tmp = X[1:, :]
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
            self.word_embeddings_vector_file = kwargs['word_emb_file']
        else:
            self.word_embeddings_vector_file = './data_preserved/word.vectors.txt'

        if 'word_emb_vocab' in kwargs:
            self.word_embeddings_dict_file = kwargs['word_emb_vocab']
        else:
            self.word_embeddings_dict_file = './data_preserved/word.vocab.txt'

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
        :param news_headline:
        :return:
        """
        news_tokenized = tokenize_sentence(news_headline,
                                           self.W_norm, self.vocab)
        news_rep = strToVector(news_tokenized,
                               self.rae_W1, self.rae_W2, self.rae_b)

        return news_rep

    def getNextDayReturn(self, dt=None, max_day_try=10):
        """
        :param dt: type datetime
        :param max_day_try: to skip stock breaks, default 10
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
        stock_data = stock_data[stock_data.shape[0]::-1, :]
        stock_data = featureNormalization(stock_data, mode=1)
        return stock_data

