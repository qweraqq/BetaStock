# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import re
import tushare as ts
from str2vec import *
from keras.preprocessing import sequence
from datetime import datetime, date, time, timedelta

import logging
# logging.basicConfig(level=logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Preprocess.py')


class Preprocessor(object):
    """
    A simple class to pre-process data
    Given one day's news, do:
        (1) convert into (number_news, str_vec_dim)
        (2) get last num_days' close percentage rate and turn over
        (3) get next day's close percentage rate (training mode)
    """

    def __init__(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        logger.info('init preprocessor...')
        allowed_kwargs = {'word_embeddings_vector_file',
                          'word_embeddings_dict_file',
                          'rae_parameters_dict',
                          'name',
                          'todo_args'}
        for kwarg in kwargs:
            assert kwarg in allowed_kwargs, 'keyword argument not understood' + kwarg

        if 'name' in kwargs:
            self.name = kwargs['name']

        if 'word_embeddings_vector_file' in kwargs:
            self.word_embeddings_vector_file = kwargs['word_embeddings_vector_file']
        else:
            self.word_embeddings_vector_file = './data_preserved/word.vectors.txt'

        if 'word_embeddings_dict_file' in kwargs:
            self.word_embeddings_dict_file = kwargs['word_embeddings_dict_file']
        else:
            self.word_embeddings_dict_file = './data_preserved/word.vocab.txt'

        if 'rae_parameters_dict' in kwargs:
            self.rae_parameters_dict = kwargs['rae_parameters_dict']
        else:
            self.rae_parameters_dict = './data_preserved/'

        # word embeddings, vocab ,ivocab
        self.W_norm, self.vocab, self.ivocab = \
            loadWordEmbeddings(self.word_embeddings_dict_file,
                               self.word_embeddings_vector_file)

        self.rae_W1, self.rae_W2, self.rae_b = \
            loadRaeParameters(self.rae_parameters_dict+'W1.txt',
                              self.rae_parameters_dict+'W2.txt',
                              self.rae_parameters_dict+'b.txt')

    def convertOneNewsToVec(self, news):
        """
        :param news: a string like 'lianghui shuili zhuaokai'
        :return: a distributed representation, size=(vec_dim,)
        """
        news_tokenized = tokenize_sentence(news, self.W_norm, self.vocab)
        logger.info('news_tokenized size='+str(news_tokenized.shape))
        news_representation = strToVector(news_tokenized,
                                          self.rae_W1, self.rae_W2, self.rae_b)
        logger.info('news_representation size='+str(news_representation.shape))
        return news_representation

    def convertOneDayAllNewsToVec(self, news_rep_list, max_len=200):
        """
        :param news_rep_list: a list of distributed news rep
        :param max_len: padding parameters, default 200
        :return: a matrix, size = (1, max_len, vec_dim)
        """
        r_tmp = np.vstack(news_rep_list)
        r_tmp = r_tmp[np.newaxis, :, :]
        r_tmp = sequence.pad_sequences(r_tmp, maxlen=max_len, dtype='float32')
        return r_tmp

    def getPreviousStockData(self, dt=None, td=200, max_len=200):
        """
        :param dt: date time of the news
        :param td: time delta, number days to backtrack, default 200
        :param max_len: padding parameters, default 200
        :return:
        """
        if dt == None:
            dt = datetime.today()
        dt2 = dt - timedelta(days=td)
        logger.info(self.formatDateString(dt2))
        stock_data = ts.get_hist_data('sh', start=self.formatDateString(dt2),
                                      end=self.formatDateString(dt))

        stock_data = stock_data.as_matrix(['open', 'high', 'low', 'p_change', 'volume'])
        stock_data = stock_data[stock_data.shape[0]::-1, :]
        return stock_data

    def getNextDayStockPchange(self, dt=None, max_td=7):
        """
        :param dt:
        :param max_td:
        :return:
        """
        if dt == None:
            return None
        dt1 = dt + timedelta(1)
        dt2 = dt + timedelta(max_td)

        stock_data = ts.get_hist_data('sh', start=self.formatDateString(dt1),
                                      end=self.formatDateString(dt2))
        if stock_data.empty:
            return None
        return stock_data.as_matrix(['p_change'])[-1]

    def formatDateString(self, dt):
        """
        :param dt:
        :return:
        """
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

if __name__ == '__main__':
    p = Preprocessor()
    # news_rep1 = p.convertOneNewsToVec("平安称陆金所下半年启动上市 不受战新板不确定性影响")
    # news_rep2 = p.convertOneNewsToVec("财经观察：全球经济风险致美联储暂缓加息")
    #
    # print news_rep1[0:5]
    # print news_rep2[0:5]
    #
    # news_rep_list = np.vstack((news_rep1, news_rep2))
    # print p.convertOneDayAllNewsToVec(news_rep_list)
    # print p.convertOneDayAllNewsToVec(news_rep_list).shape
    m = p.getPreviousStockData(td=2)
    print m
