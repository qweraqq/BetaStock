# -*- coding: utf-8 -*-
from __future__ import division
import jieba
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def loadRaeParameters(W1_filename, W2_filename, b_filename):
    """
    load trained parameters from txt files
    """
    W1 = np.loadtxt(W1_filename)
    W2 = np.loadtxt(W2_filename)
    b = np.loadtxt(b_filename)
    return W1, W2, b

    
def strToVector(tokenized_str, W1, W2, b):
    """
    convert a news_headline to distributed representation
    (just like word2vec)

    Input: a tokenized string, actually a matrix of float
           size(matirx) = sentence length, word embeddings dim
           you can use tokenizeSentence() API to tokenize
    """
    vector_dim = tokenized_str.shape[1]
    h = np.zeros(vector_dim)  # init hidden
    sentence_len = tokenized_str.shape[0]
    for x in range(sentence_len):
        h = np.tanh((1/(x+1))*np.dot(tokenized_str[x, :], W1) +
                    (x/(x+1))*np.dot(h, W2))
        h = h/(np.sum(h**2)**0.5)  # normalization
        if np.isnan(h[0]) or np.isnan(h[1]):  # special case
            h = np.zeros(vector_dim)
    return h

    
def loadWordEmbeddings(vocab_file, vectors_file):
    with open(vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    
    with open(vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]
            # vectors['中国']=[0.5,0.4,...0.1]
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}
    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v
    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return W_norm, vocab, ivocab


def get_word_distance(w1, w2, W_norm, vocab):
    """
    cosine distance between two words w1&w2
    """
    v1 = W_norm[vocab[w1], :]
    v2 = W_norm[vocab[w2], :]
    dist = np.dot(v1, v2.T)
    return dist


def show_nearest_words(w, W_norm, vocab, ivocab, n=10):
    """
    print top n(default 10) nearest words to word w
    """
    v =  W_norm[vocab[w], :]
    distances = np.dot(v, W_norm.T)
    orders = np.argsort(-distances)
    for i in range(n):
        print i, ivocab[orders[i]], distances[orders[i]]
        # print i, ivocab[orders[i]].decode('utf8').encode('gbk'), distances[orders[i]]


def tokenize_sentence(line, W_norm, vocab, tmp_lambda=0.6):
    """
    convert a sentence to a matrix && Chinese word seg revision
    matrix size = sentence length, word embeddings dim

    Input: line is the sentence(type sring) to tokenize
           tmp_lambda(type float) is a word seg revision parameter
    """
    vector_dim = W_norm.shape[1]
    rvalue = np.zeros((1,vector_dim))
    sen = list(jieba.cut(line))
    for j,w in enumerate(sen):
        w=w.encode('utf-8')
        if vocab.has_key(w)==False:
            if j>=1:
                rvalue=np.vstack((rvalue, Wj_1))
            Wj_1 = np.zeros((1, vector_dim))  # word 'unk'=[0]
            continue

        if j == 0:
            Wj_1 = np.array(W_norm[vocab[w], :])
            continue
        
        Wj = np.array(W_norm[vocab[w], :])
        if np.dot(Wj_1, Wj)>tmp_lambda:  # merge word
            np.add(Wj, Wj_1, Wj_1)
            Wj_1 = (Wj_1.T / (np.sum(Wj_1 ** 2) ** 0.5)).T
        else:
            rvalue = np.vstack((rvalue, Wj_1))
            Wj_1 = Wj

        if j == len(sen)-1:  # end
           rvalue = np.vstack((rvalue, Wj_1))
    return np.delete(rvalue, 0, 0)


if __name__ == '__main__':
    # load dataset to tokenize
    # data_file = 'ths_news_2015.txt'
    # load word embeddings
    vocab_file = 'jieba_ths_vocab_big.txt'
    vectors_file = 'jieba_ths_vectors_big.txt'
    W_norm, vocab, ivocab = loadWordEmbeddings(vocab_file, vectors_file)
    print 'show nearest words'
    show_nearest_words('上海', W_norm, vocab, ivocab, 10)
    print 'show word dist'
    print get_word_distance('上海', '北京', W_norm, vocab)
    print 'tokenize sentence'
    tokenized_str = tokenize_sentence("“双十一”仅3成商品下调价格 过半商家先涨后降", W_norm, vocab)
    print tokenized_str.shape
    
    print 'tokenized string to vector'
    W1, W2, b = loadRaeParameters('W1.txt', 'W2.txt', 'b.txt')
    print strToVector(tokenized_str, W1, W2, b)
