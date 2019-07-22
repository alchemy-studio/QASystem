#!/bin/bash

mkdir models
cd models
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip chinese_L-12_H-768_A-12.zip
rm chinese_L-12_H-768_A-12.zip

