# -*- coding: UTF-8 -*-
import os
from rae_representation_generation import *
from datetime import datetime, timedelta
from keras.preprocessing import sequence
import tushare as ts
import numpy as np
import pandas as pd
import re


__author__ = 'shenxiangxiang@gmail.com'


def get_date_return(dt=None, max_day_try=10):
    """
    given a date, return the change value of date dt
    :param dt: type datetime
    :param max_day_try: type int, to skip stock breaks, default 10
    :return: None if invalid, return_next_day otherwise
    """
    if type(dt) is not datetime:
        return None
    assert max_day_try >= 1, 'at least one day'

    dt1 = dt
    dt2 = dt + timedelta(days=max_day_try)
    stock_data = ts.get_hist_data('sh', start=formatDateString(dt1),
                                  end=formatDateString(dt2), retry_count=10)
    if stock_data.empty:
        return None
    return stock_data.as_matrix(['p_change'])[-1]
    # since the return value is reversed ordered


def get_stock_date(dt):
    """
    return the actual return date of date dt
    #Input: date time type dt
    """
    if dt.hour < 15: # 当天交易日
        dt = datetime(dt.year, dt.month, dt.day, 0, 0)
    else: # 下一交易日
        dt = datetime(dt.year, dt.month, dt.day, 0, 0) + timedelta(days=1)
    return dt

vocab_file = 'jieba_ths_vocab_big.txt'
vectors_file = 'jieba_ths_vectors_big.txt'
W_norm, vocab, ivocab = load_word_embeddings(vocab_file, vectors_file)
W1, W2, b = load_rae_parameters("W1.txt", "W2.txt", "b.txt")
def single_news_to_representation(news):
    tokenized_str, sen_cut = tokenize_sentence(news, W_norm, vocab, 0.7)
    h,_ = str_to_vector(tokenized_str, W1, W2, b)
    return h


def news_input_sequence_preprocess(one_day_output, maxlen=200):
    X_tmp = np.vstack(one_day_output)
    X_tmp = X_tmp[np.newaxis, :, :]
    X_tmp = sequence.pad_sequences(X_tmp, maxlen=maxlen, dtype='float32')
    return X_tmp


def get_news_representation(news_filename, max_len=200, year='2015年'):
    """
    Convert a file of news headlines to (X, y) tuple
    (X, y) is feed into financial news analysis model
    #Input: news file_name
    """
    base_time = datetime(1991, 12, 20, 0, 0)
    one_day_output = []
    X = None
    y = None
    with open(news_filename, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) < 1:  # skip empty line
                continue
            if re.match(r'^\d+月\d+日 \d+:\d+$', line):
                line_time = datetime.strptime(year+line, '%Y年%m月%d日 %H:%M')
                line_time = get_stock_date(line_time)
                if (line_time - base_time).days >= 1 and len(one_day_output) == 0:
                    base_time = line_time
                    one_day_output.append(single_news_to_representation(news_headline))
                elif (line_time-base_time).days >= 1 and len(one_day_output) > 0:
                    y_tmp = [[get_date_return(line_time)]]
                    y_tmp = sequence.pad_sequences(y_tmp, maxlen=1, dtype='float32')
                    X_tmp = news_input_sequence_preprocess(one_day_output, max_len)

                    X = X_tmp if X is None else np.vstack((X, X_tmp))
                    y = y_tmp if y is None else np.vstack((y, y_tmp))

                    base_time = line_time
                    one_day_output = []
                    one_day_output.append(single_news_to_representation(news_headline))
                else:
                    one_day_output.append(single_news_to_representation(news_headline))
            else:
                news_headline = line
    if X is not None: # last line
        y_tmp = [[get_date_return(base_time)]]
        y_tmp = sequence.pad_sequences(y_tmp, maxlen=1, dtype='float32')
        y = np.vstack((y, y_tmp))
        X_tmp = news_input_sequence_preprocess(one_day_output, max_len)
        X = np.vstack((X, X_tmp))

    return X,y


def formatDateString(dt):
    """
    :param dt: type datetime
    :return: formatted datetime string, such as "2015-01-01"
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

if __name__ == '__main__':
    line_time = datetime.strptime("2015年2月1日 11:31", '%Y年%m月%d日 %H:%M')
