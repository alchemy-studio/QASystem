#!/usr/bin/python3

import os;
import tensorflow as tf;
from bert import BertModelLayer;
from bert.loader import StockBertConfig, load_stock_weights;
from bert.tokenization import FullTokenizer;

def BERT(max_seq_len = 128, bert_model_dir = 'models/chinese_L-12_H-768_A-12'):

    # load bert parameters
    with tf.io.gfile.GFile(os.path.join(bert_model_dir, "bert_config.json"), "r") as reader:
        stock_params = StockBertConfig.from_json_string(reader.read());
        bert_params = stock_params.to_bert_model_layer_params();
    # create bert structure according to the parameters
    bert = BertModelLayer.from_params(bert_params, name = "bert");
    # inputs
    input_token_ids = tf.keras.Input((max_seq_len,));
    input_segment_ids = tf.keras.Input((max_seq_len,));
    # outputs
    output = bert([input_token_ids, input_segment_ids]);
    # create model containing only bert layer
    model = tf.keras.Model(inputs = [input_token_ids, input_segment_ids], outputs = output);
    model.build(input_shape = [(None, max_seq_len), (None, max_seq_len)]);
    # load bert layer weights
    load_stock_weights(bert, os.path.join(bert_model_dir, "bert_model.ckpt"));
    # create tokenizer, chinese character needs no lower case.
    tokenizer = FullTokenizer(vocab_file = os.path.join(bert_model_dir, "vocab.txt"), do_lower_case = False);
    return model, tokenizer;

