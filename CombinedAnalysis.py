# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Merge, LSTM, Dense
from keras.models import model_from_json
from str2vec import *

import keras
import numpy as np

import logging
# logging.basicConfig(level=logging.CRITICAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Combined Analysis.py')

class CombinedAnalysis(object):
    """
    Combine fundamental analysis and technical analysis for index
    Todo: Combine index and individual stock
    """

    def __init__(self, **kwargs):
        """

        :param kwargs:
        :return:
        """

        allowed_kwargs = {'index_model_file',
                          'fundamental_model_file',
                          'word_embeddings_vector_file',
                          'word_embeddings_dict_file',
                          'rae_parameters_dict',
                          'name',
                          'todo_args'}

        for kwarg in kwargs:
            assert kwarg in allowed_kwargs, 'keyword argument not understood' + kwarg

        if 'name' in kwargs:
            self.name = kwargs['name']

        if 'index_model_file' in kwargs:
            self.index_model_file = kwargs['index_model_file']
        else:
            self.index_model_file = './data_preserved/sh.model'

        if 'fundamental_model_file' in kwargs:
            self.fundamental_model_file = kwargs['fundamental_model_file']
        else:
            self.fundamental_model_file = './data_preserved/fa.model'

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

        self.index_model = model_from_json(open(self.index_model_file+'.json').read())
        self.index_model.load_weights(self.index_model_file+'.weights.h5')
        self.fundamental_model = model_from_json(open(self.fundamental_model_file+'.json').read())
        self.fundamental_model.load_weights(self.fundamental_model_file+'.weights.h5')
        self.fundamental_model_weights = self.fundamental_model.get_weights()

        # TODO: building graph model
        self.model = Sequential()

    def technicalAnalysisSingleNews(self, news):
        """
        judge whether the news is a good or bad one
        :param news: type string, like 'lianghui shunli zhaokai'
        :return: a float value, > 0 means good while < 0 means bad
        """

        news_tokenized = tokenize_sentence(news, self.W_norm, self.vocab)

        news_representation = strToVector(news_tokenized, self.rae_W1, self.rae_W2, self.rae_b)
        Wxc = self.fundamental_model_weights[3]
        bc = self.fundamental_model_weights[5]
        W = self.fundamental_model_weights[12]
        b = self.fundamental_model_weights[13]
        news_deep_rep = np.tanh(np.dot(news_representation, Wxc) + bc)
        return np.dot(news_deep_rep, W)+b

    def buildCombinedModels(self):
        """
        TODO: a tough one
        :return:
        """
        model1 = Sequential()
        model1.add(LSTM(50, input_dim=5, return_sequences=True))
        model1.add(LSTM(50, input_dim=50, return_sequences=True))
        model1.set_weights(self.index_model.get_weights())

        model2 = Sequential()
        model2.add(LSTM(100, input_dim=100, activation='tanh', inner_activation='sigmoid'))
        model2.set_weights(self.fundamental_model_weights)

        self.model.add(Merge([model1, self.model2],
                       mode='concat', concat_axis=-1))
        self.model.add(Dense(200, input_dim=150, activation='tanh'))
        self.model.add(Dense(1, input_dim=200, activation='linear'))

        print self.model.summary()


if __name__ == '__main__':
    ca = CombinedAnalysis()
    print ca.technicalAnalysisSingleNews("快讯：沪指午后发力站上2900点 创业板涨超5%")
    print ca.technicalAnalysisSingleNews("财经观察：全球经济风险致美联储暂缓加息")
    print ca.technicalAnalysisSingleNews("中石油再遭低油价重击 大庆油田前两个月亏损超50亿")
    print ca.technicalAnalysisSingleNews("深港通力争今年开通 创业板股票将纳入标的")
    print ca.technicalAnalysisSingleNews("首家自贸区合资券商申港证券获批 证券行业对外开放提速")
    print ca.technicalAnalysisSingleNews("平安称陆金所下半年启动上市 不受战新板不确定性影响")
    ca.buildCombinedModels()
