#!/usr/bin/python3

import os;
import csv;
import tensorflow as tf;
from BERT import BERT;

class Predictor(object):

    def __init__(self, max_seq_len = 128):

        # create bert object
        self.bert, self.tokenizer = BERT(max_seq_len = max_seq_len);

    def read_tsv(self, inputfile):

        with tf.io.gfile.GFile(inputfile, "r") as f:
            reader = csv.reader(f, delimiter = '\t');
            lines = [];
            for line in reader:
                lines.append(line);
            return lines;

    def create_dataset(self, examples):

        dataset = [];
        for line in examples:
            label = line[0];
            question = line[1];
            answer = line[2];
            dataset.append((question,answer,label));
        return dataset;

    def get_examples(self, data_dir = None):

        assert type(data_dir) is str;
        train_examples = self.read_tsv(os.path.join(data_dir, "train.tsv"));
        test_examples = self.read_tsv(os.path.join(data_dir, "test.tsv"));
        trainset = self.create_dataset(train_examples);
        testset = self.create_dataset(test_examples);
        return trainset, testset;

    def train(self, data_dir = None, batch = 32, epoch = 3):

        assert type(data_dir) is str;
        trainset, testset = self.get_examples(data_dir);
        num_train_steps = int(len(trainset) / batch * epoch);
        num_warmup_steps = int(num_train_steps * 0.1);
        # TODO

if __name__ == "__main__":

    assert tf.executing_eagerly();
    predictor = Predictor();
    tokens = predictor.tokenizer.tokenize('你好，世界！');
    print(len(tokens))

