#!/usr/bin/python3

import tensorflow as tf;
from BERT import BERT;
from bert.tokenization import FullTokenizer;

class Predictor(object):

    def __init__(self, max_seq_len = 128):

        # create bert object
        self.bert, self.tokenizer = BERT(max_seq_len = max_seq_len);

if __name__ == "__main__":

    assert tf.executing_eagerly();
    predictor = Predictor();
    tokens = predictor.tokenizer.tokenize('你好，世界！');
    print(len(tokens))

