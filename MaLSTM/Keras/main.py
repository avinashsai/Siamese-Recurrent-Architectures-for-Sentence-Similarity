from __future__ import print_function
import os
import re
import numpy as np 
import pandas as pd 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import argparse

from data_loader import *

from tokenizer import *

from model import *
from train import *

def main():

	parser = argparse.ArgumentParser(description='Enter hyperparameters')

	parser.add_argument('-w','--Number of words',type=int,help='Maximum words',required=True)
	parser.add_argument('-s','--Sentence Length',type=int,help='maximum sentence Length',required=True)
	parser.add_argument('-v','--Embedding dim',type=int,help='Vector Size',required=True)
	parser.add_argument('-t','--Embedding layer training',type=bool,help='To train Embedding layer',required=True)
	parser.add_argument('-l','--Hidden Layer size',type=int,help='Hidden Layer Size of LSTM',required=True)
	parser.add_argument('-b','--batchsize',type=int,help='Batch Size',required=True)
	parser.add_argument('-e','--epochs',type=int,help='Model Epochs',required=True)


	args = vars(parser.parse_args())

	maxwords = args['Number of words']
	sentence_length = args['Sentence Length']
	embedding_dim = args['Embedding dim']
	to_train = args['Embedding layer training']
	hidden_dim = args['Hidden Layer size']
	batch_size = args['batchsize']
	num_epochs = args['epochs']

	train_corpus,train_labels,valid_corpus,valid_labels,test_corpus,test_labels = load_sick_data()

	train_q1,train_q2,valid_q1,valid_q2,test_q1,test_q2 = get_indices(train_corpus,valid_corpus,test_corpus,
		maxwords,sentence_length)


	manhattan = ManhattanLSTM(hidden_dim,sentence_length,embedding_dim,to_train,maxwords)

	model = manhattan.network()

	test_accuracy = train_model(model,train_q1,train_q2,valid_q1,valid_q2,test_q1,test_q2,train_labels,valid_labels,test_labels,batch_size,num_epochs)

	print("Test Accuracy {}".format(test_accuracy))


if __name__ == '__main__':
	main()