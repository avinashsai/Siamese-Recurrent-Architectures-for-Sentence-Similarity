import sys
import re
import os
import numpy as np 
import pandas as pd 

from preprocess import *


def load_sick_data():
	sick_data = pd.read_csv('../../Datasets/SICK.csv',sep='	')

	sick_data = sick_data[['sentence_A','sentence_B','entailment_label','SemEval_set']]

	sick_data.rename(columns={'sentence_A':'sent_A','sentence_B':'sent_B',
		'entailment_label':'label','SemEval_set':'type'},inplace=True)

	sick_data =  sick_data[sick_data['label']!='NEUTRAL']
	sick_data.reset_index(drop=True,inplace=True)

	data_length = len(sick_data)

	train_corpus = [[process_text(sick_data['sent_A'][i]),process_text(sick_data['sent_B'][i])] 
	                  for i in range(data_length) if sick_data['type'][i]=='TRAIN']

	test_corpus = [[process_text(sick_data['sent_A'][i]),process_text(sick_data['sent_B'][i])] 
	                  for i in range(data_length) if sick_data['type'][i]=='TEST']

	valid_corpus = [[process_text(sick_data['sent_A'][i]),process_text(sick_data['sent_B'][i])]
	                  for i in range(data_length) if sick_data['type'][i]=='TRIAL']

	train_labels = np.array([1 if sick_data['label'][i]=='ENTAILMENT' else 0 if sick_data['label'][i]=='CONTRADICTION' else 2 
		for i in range(data_length) if sick_data['type'][i]=='TRAIN'])

	test_labels = np.array([1 if sick_data['label'][i]=='ENTAILMENT' else 0 if sick_data['label'][i]=='CONTRADICTION' else 2 
		for i in range(data_length) if sick_data['type'][i]=='TEST'])

	valid_labels = np.array([1 if sick_data['label'][i]=='ENTAILMENT' else 0 if sick_data['label'][i]=='CONTRADICTION' else 2 
		for i in range(data_length) if sick_data['type'][i]=='TRIAL'])

	del sick_data

	return train_corpus,train_labels,valid_corpus,valid_labels,test_corpus,test_labels
	
