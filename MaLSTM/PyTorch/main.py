from __future__ import print_function
import os
import re
import numpy as np 
import pandas as pd 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import argparse

import torch
import torch.utils.data

from loader import *
from generator import *

from model import *
from train import *

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
batchsize = args['batchsize']
num_epochs = args['epochs']


train_corpus,train_labels,valid_corpus,valid_labels,test_corpus,test_labels = load_sick_data()

train_ind1,train_ind2,valid_ind1,valid_ind2,test_ind1,test_ind2 = generate_indices(train_corpus,valid_corpus,
	test_corpus,sentence_length,maxwords)


lstm_model = lstm(embedding_dim,hidden_dim,batchsize,maxwords)

out = lstm_model(train_ind1[:batchsize,:],train_ind2[:batchsize,:])
print(out.shape)

train_model(lstm_model,train_ind1,train_ind2,valid_ind1,valid_ind2,test_ind1,test_ind2,train_labels,
	valid_labels,test_labels,batchsize,num_epochs)