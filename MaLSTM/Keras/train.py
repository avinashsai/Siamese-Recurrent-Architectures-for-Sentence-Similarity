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



def train_model(model,train_q1,train_q2,valid_q1,valid_q2,test_q1,test_q2,train_labels,valid_labels,test_labels,batchsize,epochs):

	model.fit([train_q1,train_q2],train_labels,validation_data=([valid_q1,valid_q2],valid_labels),batch_size=batchsize,epochs=epochs)

	preds = model.predict([test_q1,test_q2])

	ypred = (preds>=0.5).reshape(preds.shape[0])

	return (np.sum(ypred==test_labels)/(ypred.shape[0]))*100


