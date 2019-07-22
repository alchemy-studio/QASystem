#!/usr/bin/python3

import tensorflow as tf;
from BERT import BERT;

class Predictor(object):

    def __init__(self, max_seq_len = 128):

        # create bert object
        self.bert = BERT(max_seq_len = max_seq_len);

if __name__ == "__main__":

    assert tf.executing_eagerly();
    predictor = Predictor();

