#!/usr/bin/python3

import os;
import csv;
import tensorflow as tf;
from BERT import BERT;

class Predictor(object):

    def __init__(self, max_seq_len = 128):

        # create bert object and tokenizer
        self.bert, self.tokenizer = BERT(max_seq_len = max_seq_len);
        # save max sequence length
        self.max_seq_len = max_seq_len;

    def _read_tsv(self, inputfile):

        with tf.io.gfile.GFile(inputfile, "r") as f:
            reader = csv.reader(f, delimiter = '\t');
            lines = [];
            for line in reader:
                lines.append(line);
            return lines;

    def _create_classifier_examples(self, examples):

        dataset = [];
        for line in examples:
            label = int(line[0]);
            question = line[1];
            answer = line[2];
            dataset.append((question,answer,label));
        return dataset;

    def create_classifier_datasets(self, data_dir = None):

        assert type(data_dir) is str;
        train_examples = self._read_tsv(os.path.join(data_dir, "train.tsv"));
        test_examples = self._read_tsv(os.path.join(data_dir, "test.tsv"));
        trainset = self._create_classifier_examples(train_examples);
        testset = self._create_classifier_examples(test_examples);
        write_tfrecord(trainset, "trainset.tfrecord");
        write_tfrecord(testset, "testset.tfrecord");
        def write_tfrecord(dataset, output_file):
            # write to tfrecord
            writer = tf.io.TFRecordWriter(output_file);
            for example in dataset:
                # tokenize question and answer.
                tokens_question = self.tokenizer.tokenize(example[0]);
                tokens_answer = self.tokenizer.tokenize(example[1]);
                # truncate to max seq len.
                while True:
                    total_length = len(tokens_question) + len(tokens_answer);
                    if total_length <= self.max_seq_len: break;
                    if len(tokens_question) > len(tokens_answer): tokens_question.pop();
                    else: tokens_answer.pop();
                tokens = [];
                segment_ids = [];
                # insert question segment
                tokens.append('[CLS]');
                segment_ids.append(0);
                for token in tokens_question:
                    tokens.append(token);
                    segment_ids.append(0);
                tokens.append('[SEP]');
                segment_ids.append(0);
                # insert answer segment
                for token in tokens_answer:
                    tokens.append(token);
                    segment_ids.append(1);
                tokens.append('[SEP]');
                segment_ids.append(1);
                # tokenize into input_ids
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens);
                # mask the valid token
                input_mask = [1] * len(input_ids);
                # padding 0
                while len(input_ids) < self.max_seq_len:
                    input_ids.append(0);
                    input_mask.append(0);
                    segment_ids.append(0);
                assert len(input_ids) == self.max_seq_len;
                assert len(input_mask) == self.max_seq_len;
                assert len(segment_ids) == self.max_seq_len;
                # write to tfrecord
                tf_example = tf.train.Example(features = tf.train.Features(
                    feature = {
                        "input_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_ids))),
                        "input_mask": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_mask))),
                        "segment_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(segment_ids))),
                        "label_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(example[2])))
                    }
                ));
                writer.write(tf_example.SerializeToString());
            writer.close();

    @tf.function
    def classifier_input_fn(self, serialized_example):

        feature = tf.io.parse_single_example(
            serialized_example,
            features = {
                'input_ids': tf.io.FixedLenFeature((self.max_seq_len), dtype = tf.int64),
                'input_mask': tf.io.FixedLenFeature((self.max_seq_len), dtype = tf.int64),
                'segment_ids': tf.io.FixedLenFeature((self.max_seq_len), dtype = tf.int64),
                'label_ids': tf.io.FixedLenFeature((), dtype = tf.int64)
            }
        );
        for name in list(feature.keys()):
            feature[name] = tf.cast(feature[name], dtype =tf.int32);
        return feature;

    def train_classifier(self, data_dir = None, batch = 32, epoch = 3):

        assert type(data_dir) is str;
        # create dataset in tfrecord format.
        self.create_classifier_datasets(data_dir);
        # load from the tfrecord file
        trainset = tf.data.TFRecordDataset('trainset.tfrecord').map(self.classifier_input_fn).repeat().shuffle(buffer_size = 100);
        # finetune the bert model
        optimizer = tf.keras.optimizer.Adam(2e-5);
        log = tf.summary.create_file_writer('classifier');
        avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
        for epoch in range(3):
            for features in datasets:
                with tf.GradientTape() as tape:
                    logits = self.classify([features['input_ids'], features['segment_ids']], features['input_mask']);
                    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True)(features['label_ids'], logits);
                avg_loss.update_state(loss);
                # write log
                if tf.equal(optimizer.iterations % 100, 0):
                    with log.as_default():
                        tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
                    print('Step #%d Loss: %.6f' % (optimizer.iterations, avg_loss.result()));
                    avg_loss.reset_states();
                grads = tape.gradient(loss, self.bert.trainable_variables);
                optimizer.apply_gradients(zip(grads, self.bert.trainable_variables));
            # save model once every epoch
            self.bert.save('classifer/bert_%d.h5' % optimizer.iterations);

    @tf.function
    def classify(self, inputs, mask):

        # the first element of output sequence.
        outputs = self.bert(inputs, mask, True);
        # first_token.shape = (batch, hidden_size)
        first_token = outputs[:,0,:];
        pooled_output = tf.keras.layers.Dense(units = first_token.shape[-1], activation = tf.math.tanh)(first_token);
        dropout = tf.keras.layers.Dropout(rate = 0.1)(pooled_output);
        logits = tf.keras.layers.Dense(units = 2)(dropout);

        return logits;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    predictor = Predictor();
    tokens = predictor.tokenizer.tokenize('你好，世界！');
    print(len(tokens))

