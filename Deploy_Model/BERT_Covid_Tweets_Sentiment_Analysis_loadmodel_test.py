# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:59:24 2021

@author: jvillanuev29
"""

import pandas as pd
import tensorflow as tf
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
import os
from BERT_Covid_Tweets_Sentiment_Analysis import TextData

#Read data
bert_tf_testdata = pd.read_csv('Bert_test_data.csv')
bert_tf_testdata = bert_tf_testdata.dropna(how='any')
bert_tf_traindata = pd.read_csv('Bert_train_data.csv')
bert_tf_testdata = bert_tf_testdata.dropna(how='any')

#Get tokenized data
model_name = "uncased_L-12_H-768_A-12"
model_dir = bert.fetch_google_bert_model(model_name, ".models")
tokenizer = FullTokenizer(
  vocab_file = os.path.join(model_dir, "vocab.txt")
)
test_data = TextData(data=bert_tf_testdata,DATA_COLUMN='OriginalTweet',LABEL_COLUMN='Sentiment',
                     tokenizer = tokenizer, classes=bert_tf_traindata.Sentiment.unique().tolist(),
                     max_seq_len=128)
#Load model and use on data
loaded_model = tf.keras.models.load_model('./BERT_covid')
_, test_accuracy = loaded_model.evaluate(test_data.data_x, test_data.data_y)

