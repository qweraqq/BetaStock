# -*- coding: utf-8 -*-
from keras.models import Sequential, Graph
from keras.layers import Merge, LSTM, Dense, Dropout, AutoEncoder
from keras.models import model_from_json
from str2vec import *

import keras
import numpy as np
from Preprocess import Preprocessor

import logging
logging.basicConfig(level=logging.CRITICAL)
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Combined Analysis.py')

import theano.tensor as T

def custom_objective(y_true, y_pred,):
    """
    Custom objective function
    :param y_true: real value
    :param y_pred: predicted value
    :return: cost
    """
    on_unused_input='warn'
    y1 = T.as_tensor_variable(y_pred)
    y2 = T.as_tensor_variable(y_true)
    return 0*y1+0*T.sum(y2)

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
        # self.model = Graph()

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
        model1.add(LSTM(50, input_shape=(200, 5), return_sequences=True))
        model1.add(LSTM(50, input_dim=50))
        model1.set_weights(self.index_model.get_weights())

        model2 = Sequential()
        model2.add(LSTM(100, input_shape=(200, 100), activation='tanh', inner_activation='sigmoid'))
        model2.set_weights(self.fundamental_model_weights)
        self.model.add(Merge([model1, model2],
                       mode='concat', concat_axis=-1))
        # self.model.add(Dense(100, input_dim=150, activation='tanh'))
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(1, input_dim=150, activation='linear'))

        print self.model.summary()


    # def buildCombinedGraphModels(self):
    #     """
    #     TODO: a tough one
    #     :return:
    #     """
    #     self.model.add_input(name='news', input_shape=(200, 100))
    #     self.model.add_input(name='stock', input_shape=(200, 5))
    #     self.model.add_node(LSTM(100, input_shape=(200, 100), activation='tanh', inner_activation='sigmoid'),
    #                         name='fa', input='news')
    #     self.model.add_node(LSTM(50, input_shape=(200, 5), return_sequences=True),
    #                         name='ta1', input='stock')
    #     self.model.add_node(LSTM(LSTM(50, input_dim=50)), name='ta2', input='ta1')
    #     self.model.add_output(name='combined', inputs=['fa', 'ta2'],
    #                           merge_mode='concat')
    #     self.model.add_node(Dense(1), input='combined')
    #     print self.model.summary()

if __name__ == '__main__':
    # ca = CombinedAnalysis()
    # print ca.technicalAnalysisSingleNews("快讯：沪指午后发力站上2900点 创业板涨超5%")
    # print ca.technicalAnalysisSingleNews("财经观察：全球经济风险致美联储暂缓加息")
    # print ca.technicalAnalysisSingleNews("中石油再遭低油价重击 大庆油田前两个月亏损超50亿")
    # print ca.technicalAnalysisSingleNews("深港通力争今年开通 创业板股票将纳入标的")
    # print ca.technicalAnalysisSingleNews("首家自贸区合资券商申港证券获批 证券行业对外开放提速")
    # print ca.technicalAnalysisSingleNews("平安称陆金所下半年启动上市 不受战新板不确定性影响")
    # ca.buildCombinedModels()
    preprocess = Preprocessor()
    # X1, X2, y =preprocess.readNewsFromFile('training_new.txt', max_len=200)
    X1 = np.load('X1.txt')
    X2 = np.load('X2.txt')
    y = np.load('y.txt')
    # print X1.shape
    # print X2.shape
    # print y.shape
    X3 = np.load('X3.txt')
    print X3.shape

    # ca.model.compile(loss=custom_objective, optimizer='adagrad')
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=20, verbose=0, mode='auto')
    # ca.model.fit([X2, X1], y, batch_size=100, nb_epoch=1, validation_split=0.1,
    #              callbacks=[earlyStopping], shuffle=True)
    # X3 = ca.model.predict([X2, X1])
    # np.save(open('X3.txt', 'w'), X3)
    # json_string = ca.model.to_json()
    # f = open("ca.model.json", "w")
    # f.write(json_string)
    # f.close()
    # ca.model.save_weights('ca.model.weights.h5')

    # np.save(open('X2.txt', 'w'), X2)
    # np.save(open('X1.txt', 'w'), X1)
    # np.save(open('y.txt', 'w'), y)

    encoder = Sequential([Dense(150, input_dim=150, activation='tanh'), Dropout(0.2)])
    decoder = Sequential([Dense(150, input_dim=150, activation='linear')])
    ae = Sequential()
    ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
                       output_reconstruction=False))
    ae.compile(loss='mean_squared_error', optimizer='rmsprop')
    ae.fit(X3, X3, batch_size=100, nb_epoch=200)

    final_layer = Sequential()
    final_layer.add(ae.layers[0].encoder)
    final_layer.add(Dropout(0.2))
    final_layer.add(Dense(1, input_dim=150, activation='linear'))
    final_layer.compile(loss='mean_squared_error', optimizer='rmsprop')
    final_layer.fit(X3, y, validation_split=0.5,
                    callbacks=[earlyStopping], shuffle=True)

    y_predcit = final_layer.predict(X3)
    for idx,yy in enumerate(y):
        print y[idx], y_predcit[idx]