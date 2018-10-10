import sys
import re
import numpy as np 

import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

def tokenize(maxwords):
	tokenizer = Tokenizer(num_words=maxwords)
	return tokenizer



def get_indices(train_corpus,valid_corpus,test_corpus,maxwords,sentence_length):

	train_q1 = [sent[0] for sent in train_corpus]
	train_q2 = [sent[1] for sent in train_corpus]

	valid_q1 = [sent[0] for sent in valid_corpus]
	valid_q2 = [sent[1] for sent in valid_corpus]

	test_q1 = [sent[0] for sent in test_corpus]
	test_q2 = [sent[1] for sent in test_corpus]

	tokenizer = tokenize(maxwords)
	tokenizer.fit_on_texts(train_q1+train_q2)

	train_ind1 = pad_sequences(tokenizer.texts_to_sequences(train_q1),maxlen=sentence_length)
	train_ind2 = pad_sequences(tokenizer.texts_to_sequences(train_q2),maxlen=sentence_length)

	valid_ind1 = pad_sequences(tokenizer.texts_to_sequences(valid_q1),maxlen=sentence_length)
	valid_ind2 = pad_sequences(tokenizer.texts_to_sequences(valid_q2),maxlen=sentence_length)

	test_ind1 = pad_sequences(tokenizer.texts_to_sequences(test_q1),maxlen=sentence_length)
	test_ind2 = pad_sequences(tokenizer.texts_to_sequences(test_q2),maxlen=sentence_length)


	return train_ind1,train_ind2,valid_ind1,valid_ind2,test_ind1,test_ind2



