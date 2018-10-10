import sys
import os
import re
import numpy as np

import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge,Bidirectional
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU,Dense,Dropout,Lambda
from keras import metrics
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate


class ManhattanLSTM:
	def __init__(self,hiddendim,maxsentence_length,embedding_dim,is_trainable,max_num_words):
		self.hiddendim = hiddendim
		self.maxsentence_length = maxsentence_length
		self.embedding_dim = embedding_dim
		self.is_trainable = is_trainable
		self.max_num_words = max_num_words

	def manhattandistance(self,l1,l2):
		return K.exp(-K.sum(K.abs(l1-l2), axis=1, keepdims=True))

	def network(self):

		ques1 = Input(shape=(self.maxsentence_length,))
		ques2 = Input(shape=(self.maxsentence_length,))

		embedding_layer = Embedding(input_dim=self.max_num_words,output_dim=self.embedding_dim,
     		trainable=self.is_trainable,input_length=self.maxsentence_length)

		ques1_embed = embedding_layer(ques1)
		ques2_embed = embedding_layer(ques2)

		lstm = LSTM(self.hiddendim,return_sequences=False)

		ques1_lstm_out = lstm(ques1_embed)
		ques2_lstm_out = lstm(ques2_embed)

		manhattan_dis = Lambda(lambda x:self.manhattandistance(x[0],x[1]),output_shape=lambda x:(x[0][0],1))([ques1_lstm_out,ques2_lstm_out])

		model = Model(inputs=[ques1,ques2],outputs=manhattan_dis)

		optimizer = Adadelta(clipnorm=1.25)

		model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

		return model

