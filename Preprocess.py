# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import re
import tushare as ts
from pip.utils.ui import hidden_cursor

from str2vec import *
from keras.preprocessing import sequence
from datetime import datetime, date, time, timedelta
import logging
logging.basicConfig(level=logging.CRITICAL)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Preprocess.py')


def featureNormalization(X, mode=0):
    """
    :param X: (nb_examples, nb_features) matrix
              here, nb_samples a.k.a timesteps
              X[: ,0] = (open-last_day_close)/last_day_close
              X[: ,1] = (high-last_day_close)/last_day_close
              X[: ,2] = (low-last_day_close)/last_day_close
              X[: ,3] = (close-last_day_close)/last_day_close
              X[: ,4] = turnover rate
              X[: ,5] = close
    :param mode: dummy now
    :return: normalized X
    """

    if mode == 1:
        X[:, 4] = X[:, 4] * 100
        X[:, 4] = X[:, 4] / 320000000000
    X[:, 3] = X[:, 3]/100
    tmp = X[1:, :]
    last_day_close = X[0:-1, 5]
    last_day_close = last_day_close.reshape((len(last_day_close), 1))
    tmp[:, 0:3] = (tmp[:, 0:3] - last_day_close) / last_day_close
    tmp[:, 0:5] = tmp[:, 0:5] * 10
    return tmp[:, 0:5]


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
        # logger.info('news_tokenized size='+str(news_tokenized.shape))
        news_representation = strToVector(news_tokenized,
                                          self.rae_W1, self.rae_W2, self.rae_b)
        # logger.info('news_representation size='+str(news_representation.shape))
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
        # logger.info(self.formatDateString(dt2))
        stock_data = ts.get_hist_data('sh', start=self.formatDateString(dt2),
                                      end=self.formatDateString(dt), retry_count=10)

        stock_data = stock_data.as_matrix(['open', 'high', 'low', 'p_change', 'volume', 'close'])
        stock_data = stock_data[stock_data.shape[0]::-1, :]
        stock_data = featureNormalization(stock_data, mode=1)
        #
        return stock_data

    def getNextDayStockPchange(self, dt=None, max_td=20):
        """
        :param dt:
        :param max_td:
        :return:
        """
        if dt == None:
            return None
        dt1 = dt + timedelta(days=1)
        dt2 = dt + timedelta(days=max_td)

        stock_data = ts.get_hist_data('sh', start=self.formatDateString(dt1),
                                      end=self.formatDateString(dt2), retry_count=10)
        if stock_data.empty:
            return 0
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

    def readNewsFromFile(self, filename, return_pchange=True,
                         max_len=200, year='2015'):
        """
        :param filename:
        :param max_len:
        :return: tensor(nb_days, max_len, vector_dim) - news vector
                 tensor(nb_days, 1) - date
        """
        one_day_news = list()
        news = ""
        base_time = datetime(1991, 12, 20, 0, 0)
        r_news = None
        r_pchange = None
        r_stock = None
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) < 1:
                    continue  # skip if empty line
                if re.match(r'^\d+月\d+日 \d+:\d+$', line):
                    line_time = datetime.strptime(year+"年"+line,
                                                  '%Y年%m月%d日 %H:%M')
                    line_time = self.getStockDate(line_time)

                    # one day data process
                    if (line_time-base_time).days >= 1 \
                            and len(one_day_news) == 0:
                        base_time = line_time
                        one_day_news.append(self.convertOneNewsToVec(news))
                    elif (line_time-base_time).days >= 1 \
                            and len(one_day_news) > 0:

                        # first, r_news
                        tmp = self.convertOneDayAllNewsToVec(one_day_news,
                                                             max_len=max_len)
                        if r_news is None:
                            r_news = tmp
                        else:
                            r_news = np.vstack((r_news, tmp))

                        # second, r_stock
                        tmp = self.getPreviousStockData(dt=base_time-timedelta(days=1),
                                                        td=150)
                        tmp = tmp[np.newaxis, :, :]
                        tmp = sequence.pad_sequences(tmp, maxlen=max_len,
                                                     dtype='float32')

                        if r_stock is None:
                            r_stock = tmp
                        else:
                            r_stock = np.vstack((r_stock, tmp))

                        # third, r_pchange
                        if return_pchange:
                            p_change = self.getNextDayStockPchange(base_time-timedelta(days=1))
                            logger.info("next day change ="+str(p_change))
                            tmp = [p_change]
                            if r_pchange is None:
                                r_pchange = tmp
                            else:
                                r_pchange = np.vstack((r_pchange, tmp))
                        else:
                            pass

                        base_time = line_time
                        one_day_news = list()
                        one_day_news.append(self.convertOneNewsToVec(news))

                        logger.info('one day++')
                    else:
                        one_day_news.append(self.convertOneNewsToVec(news))
                else:
                    news = line
        if r_news is not None:
            tmp = self.convertOneDayAllNewsToVec(one_day_news,
                                                 max_len=max_len)
            r_news = np.vstack((r_news, tmp))
            tmp = self.getPreviousStockData(dt=base_time-timedelta(days=1),
                                            td=150)
            tmp = tmp[np.newaxis, :, :]
            tmp = sequence.pad_sequences(tmp, maxlen=max_len,
                                         dtype='float32')
            r_stock = np.vstack((r_stock, tmp))

            if return_pchange:
                p_change = self.getNextDayStockPchange(base_time-timedelta(days=1))
                r_pchange = np.vstack((r_pchange, [p_change]))

        return r_news, r_stock, r_pchange

    def getStockDate(self, dt):
        """
        return the actual return date of date dt
        #Input: date time type dt
        """
        if dt.hour < 15:  # 当天交易日
            dt = datetime(dt.year, dt.month, dt.day, 0, 0)
        else:  # 下一交易日
            dt = datetime(dt.year, dt.month, dt.day, 0, 0) + timedelta(days=1)
        return dt

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
    m = p.getPreviousStockData(td=3)
    #print m
    X1,X2,y = p.readNewsFromFile('201501news.txt', max_len=3)
    print X1
    print X2[:,99:,:]
    print y
