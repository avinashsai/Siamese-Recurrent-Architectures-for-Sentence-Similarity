import sys
import re
import os
import torch

import collections
from collections import Counter


def generate_dictionary(Xtrain,max_words):

	words = []
	for sentence1,sentence2 in Xtrain:
		words+=sentence1.split()+sentence2.split()

	most_n = Counter(words).most_common(max_words)

	counts_dict = {}

	index = 1
	for word,_ in most_n:
		counts_dict[word] = index
		index+=1

	return counts_dict


def get_indices(X,max_length,counts_dict):

	data_length = len(X)
	
	ind1 = torch.zeros((data_length,max_length)).long()
	ind2 = torch.zeros((data_length,max_length)).long()

	index = 0

	for sentence1,sentence2 in X:

		sent1_toks = sentence1.split()
		sent2_toks = sentence2.split()

		sent1_toks = sent1_toks[:min(max_length,len(sent1_toks))]
		sent2_toks = sent2_toks[:min(max_length,len(sent2_toks))]


		for i in range(0,len(sent1_toks)):
			if(sent1_toks[i] in counts_dict):
				ind1[index,i] = counts_dict[sent1_toks[i]]

		for i in range(0,len(sent2_toks)):
			if(sent2_toks[i] in counts_dict):
				ind2[index,i] = counts_dict[sent2_toks[i]]

		index+=1

	return ind1,ind2


def generate_indices(Xtrain,Xvalid,Xtest,max_length,max_words):

	counts_dict = generate_dictionary(Xtrain,max_words)

	train_ind1,train_ind2 = get_indices(Xtrain,max_length,counts_dict)

	valid_ind1,valid_ind2 = get_indices(Xvalid,max_length,counts_dict)

	test_ind1,test_ind2 = get_indices(Xtest,max_length,counts_dict)

	return train_ind1,train_ind2,valid_ind1,valid_ind2,test_ind1,test_ind2