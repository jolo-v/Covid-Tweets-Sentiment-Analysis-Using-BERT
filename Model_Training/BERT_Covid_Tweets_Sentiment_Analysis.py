# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:59:21 2022

@author: jvillanuev29
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
import os

#Write class that tokenizes data
class TextData:

  def __init__(self, data, DATA_COLUMN, LABEL_COLUMN, tokenizer, classes, max_seq_len):
    self.DATA_COLUMN = DATA_COLUMN
    self.LABEL_COLUMN = LABEL_COLUMN
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    self.classes = classes
    
    self.data_x, self.data_y = self._prepare(data)

    print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.data_x = self._pad(self.data_x)

  def _prepare(self, df):
    x, y = [], []
    
    for _, row in tqdm(df.iterrows()):
      text, label = row[self.DATA_COLUMN], row[self.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)

def main():
    #Read dataset
    corona_raw = pd.read_csv('Corona_NLP_train.csv',encoding='ISO-8859-1')
    #Explore
    corona_raw.groupby('Sentiment').size()
    #Drop unnecessary columns
    corona = corona_raw[['OriginalTweet','Sentiment']]
    #Drop na
    corona = corona.dropna(how='any')#no na
    #Split into train and test
    from sklearn.model_selection import train_test_split as tts
    X_train,X_test,y_train,y_test = tts(corona['OriginalTweet'],corona['Sentiment'],
                                        test_size=0.2,random_state=32,stratify=corona['Sentiment'])
    X_train = pd.DataFrame(X_train,columns=['OriginalTweet'])
    X_train.reset_index(level=None,inplace=True,drop=True)
    y_train = pd.DataFrame(y_train,columns=['Sentiment'])
    y_train.reset_index(level=None,inplace=True,drop=True)
    X_test = pd.DataFrame(X_test,columns=['OriginalTweet'])
    X_test.reset_index(level=None,inplace=True,drop=True)
    y_test = pd.DataFrame(y_test,columns=['Sentiment'])
    y_test.reset_index(level=None,inplace=True,drop=True)
    #Collapse categories for experimentation
    sentiment_dict = {'Extremely Positive':'Positive','Positive':'Positive','Neutral':'Neutral',
                      'Extremely Negative':'Negative','Negative':'Negative'}
    y_train['Sentiment'] = y_train['Sentiment'].map(sentiment_dict)
    y_test['Sentiment'] = y_test['Sentiment'].map(sentiment_dict)

    #Load BERT and Tokenizer
    model_name = "uncased_L-12_H-768_A-12"
    model_dir = bert.fetch_google_bert_model(model_name, ".models")
    model_ckpt = os.path.join(model_dir, "bert_model.ckpt")
    #Define tokenizer
    tokenizer = FullTokenizer(
      vocab_file = os.path.join(model_dir, "vocab.txt")
    )
    #test it out
    tokens = tokenizer.tokenize("I can't wait to visit Bulgaria again!")
    tokenizer.convert_tokens_to_ids(tokens)
   
    #Create model architecture
    def create_model(max_seq_len, classes, bert_ckpt_file):
    
      bert_params = bert.params_from_pretrained_ckpt(model_dir)
      bert_params.adapter_size = None
      l_bert = BertModelLayer.from_params(bert_params, name="bert")
            
      input_ids = tf.keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
      bert_output = l_bert(input_ids)
    
      print("bert shape", bert_output.shape)
    
      cls_out = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
      cls_out = tf.keras.layers.Dropout(0.5)(cls_out)
      logits = tf.keras.layers.Dense(units=768, activation="tanh")(cls_out)
      logits = tf.keras.layers.Dropout(0.5)(logits)
      logits = tf.keras.layers.Dense(units=len(classes), activation="softmax")(logits)
    
      model = tf.keras.Model(inputs=input_ids, outputs=logits)
      model.build(input_shape=(None, max_seq_len))
    
      load_stock_weights(l_bert, bert_ckpt_file)
            
      return model

    ##Let's train the model
    #data
    bert_tf_traindata = pd.concat([X_train,y_train],axis=1)
    bert_tf_testdata = pd.concat([X_test,y_test],axis=1)
    classes = y_train.Sentiment.unique().tolist()
    train_data = TextData(data=bert_tf_traindata,DATA_COLUMN='OriginalTweet',LABEL_COLUMN='Sentiment',
                    tokenizer = tokenizer, classes=classes, max_seq_len=128)
    train_data.data_x.shape
    #model
    model = create_model(train_data.max_seq_len, classes, model_ckpt)
    model.summary()
    model.compile(
      optimizer=tf.keras.optimizers.Adam(1e-5),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )
    
    model.fit(
      x=train_data.data_x,
      y=train_data.data_y,
      validation_split=0.1,
      batch_size=16,
      shuffle=True,
      epochs=4,
    )
    #predict and evaluate train data
    from sklearn.metrics import accuracy_score
    accuracy_score(y_true=train_data.data_y,y_pred=model.predict(train_data.data_x))
    #predict and evaluate test data
    test_data = TextData(data=bert_tf_testdata,DATA_COLUMN='OriginalTweet',LABEL_COLUMN='Sentiment',
                    tokenizer = tokenizer, classes=classes, max_seq_len=128)
    ybert_pred_nn = model.predict(test_data.data_x)
    ybert_pred_nn = pd.DataFrame(ybert_pred_nn)
    ybert_pred_nn = ybert_pred_nn.idxmax(axis='columns')
    accuracy_score(y_true=test_data.data_y,y_pred=ybert_pred_nn)
    _, test_accuracy = model.evaluate(test_data.data_x, test_data.data_y)
    
    #save model and use it in other file to test
    tf.saved_model.save(model, export_dir='./BERT_covid')
    bert_tf_testdata.to_csv('Bert_test_data.csv')

if __name__ == '__main__':
    main()

