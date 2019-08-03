#!/usr/bin/python3

import os;
import csv;
import tensorflow as tf;
from BERT import BERTClassifier;

# NOTE: only member functions without under score prefix are mean for users.

class Predictor(object):

    def __init__(self, max_seq_len = 128):

        # create bert object and tokenizer
        self.classifier, self.tokenizer = BERTClassifier(max_seq_len = max_seq_len);
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

    def _create_classifier_datasets(self, data_dir = None):

        assert type(data_dir) is str;
        train_examples = self._read_tsv(os.path.join(data_dir, "train.tsv"));
        test_examples = self._read_tsv(os.path.join(data_dir, "test.tsv"));
        trainset = self._create_classifier_examples(train_examples);
        testset = self._create_classifier_examples(test_examples);
        def write_tfrecord(dataset, output_file):
            # write to tfrecord
            writer = tf.io.TFRecordWriter(output_file);
            for example in dataset:
                input_ids, input_mask, segment_ids = self._preprocess(example[0], example[1]);
                # write to tfrecord
                tf_example = tf.train.Example(features = tf.train.Features(
                    feature = {
                        "input_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_ids))),
                        "input_mask": tf.train.Feature(int64_list = tf.train.Int64List(value = list(input_mask))),
                        "segment_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = list(segment_ids))),
                        "label_ids": tf.train.Feature(int64_list = tf.train.Int64List(value = [example[2]]))
                    }
                ));
                writer.write(tf_example.SerializeToString());
            writer.close();
        write_tfrecord(trainset, "trainset.tfrecord");
        write_tfrecord(testset, "testset.tfrecord");

    def _preprocess(self, question, answer):

        # tokenize question and answer.
        tokens_question = self.tokenizer.tokenize(question);
        tokens_answer = self.tokenizer.tokenize(answer);
        # truncate to max seq len.
        while True:
            total_length = len(tokens_question) + len(tokens_answer);
            if total_length <= self.max_seq_len - 3: break;
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
        return input_ids, input_mask, segment_ids;

    @tf.function
    def _classifier_input_fn(self, serialized_example):

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
            feature[name] = tf.cast(feature[name], dtype = tf.int32);
        return (feature['input_ids'], feature['segment_ids']), feature['label_ids'];

    def finetune_classifier(self, data_dir = None, batch = 32, epochs = 3):

        assert type(data_dir) is str;
        # create dataset in tfrecord format.
        self._create_classifier_datasets(data_dir);
        # load from the tfrecord file
        trainset = tf.data.TFRecordDataset('trainset.tfrecord').map(self._classifier_input_fn).batch(batch).repeat().shuffle(buffer_size = 100);
        # finetune the bert model
        optimizer = tf.keras.optimizers.Adam(2e-5);
        self.classifier.fit(trainset, epochs = epochs, steps_per_epoch = 9);
        # save model
        self.classifier.save_weights('classifer/bert.h5');

    def predict(self, question, answer):

        input_ids, input_mask, segment_ids = self._preprocess(question, answer);
        # add batch dim.
        input_ids = tf.expand_dims(tf.constant(input_ids, dtype = tf.int32),0);
        segment_ids = tf.expand_dims(tf.constant(segment_ids, dtype = tf.int32),0);
        logits = self.classifier.predict([input_ids, segment_ids]);
        out = tf.math.argmax(logits, axis = -1)[0];
        return out;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    predictor = Predictor();
    #print(predictor.predict('今天天气如何？','感觉今天天气很不错。'));
    predictor.finetune_classifier('dataset');

