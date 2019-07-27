#!/usr/bin/python3

import os;
import tensorflow as tf;
from bert import BertModelLayer;
from bert.loader import StockBertConfig, load_stock_weights;
from bert.tokenization import FullTokenizer;

def flatten_layers(root_layer):
    if isinstance(root_layer, tf.keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer

def freeze_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False

def BERTClassifier(max_seq_len = 128, bert_model_dir = 'models/chinese_L-12_H-768_A-12', do_lower_case = False):

    # load bert parameters
    with tf.io.gfile.GFile(os.path.join(bert_model_dir, "bert_config.json"), "r") as reader:
        stock_params = StockBertConfig.from_json_string(reader.read());
        bert_params = stock_params.to_bert_model_layer_params();
    # create bert structure according to the parameters
    bert = BertModelLayer.from_params(bert_params, name = "bert");
    # inputs
    input_token_ids = tf.keras.Input((max_seq_len,), dtype = tf.int32, name = 'input_ids');
    input_segment_ids = tf.keras.Input((max_seq_len,), dtype = tf.int32, name = 'token_type_ids');
    # classifier
    output = bert([input_token_ids, input_segment_ids]);
    cls_out = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(output);
    cls_out = tf.keras.layers.Dropout(rate=0.5)(cls_out);
    logits = tf.keras.layers.Dense(units=cls_out.shape[-1], activation=tf.math.tanh)(cls_out);
    logits = tf.keras.layers.Dropout(rate=0.5)(logits);
    logits = tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)(logits);
    # create model containing only bert layer
    model = tf.keras.Model(inputs = [input_token_ids, input_segment_ids], outputs = logits);
    model.build(input_shape = [(None, max_seq_len), (None, max_seq_len)]);
    # load bert layer weights
    load_stock_weights(bert, os.path.join(bert_model_dir, "bert_model.ckpt"));
    # freeze_bert_layers
    freeze_bert_layers(bert);
    model.compile(optimizer = tf.keras.optimizers.Adam(2e-5), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name = 'accuracy')]);
    # create tokenizer, chinese character needs no lower case.
    tokenizer = FullTokenizer(vocab_file = os.path.join(bert_model_dir, "vocab.txt"), do_lower_case = do_lower_case);
    return model, tokenizer;

