import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopword = stopwords.words('english')

def process_text(text):
	text = re.sub(r"it\'s","it is",str(text))
	text = re.sub(r"i\'d","i would",str(text))
	text = re.sub(r"don\'t","do not",str(text))
	text = re.sub(r"he\'s","he is",str(text))
	text = re.sub(r"there\'s","there is",str(text))
	text = re.sub(r"that\'s","that is",str(text))
	text = re.sub(r"can\'t", "can not", text)
	text = re.sub(r"cannot", "can not ", text)
	text = re.sub(r"what\'s", "what is", text)
	text = re.sub(r"What\'s", "what is", text)
	text = re.sub(r"\'ve ", " have ", text)
	text = re.sub(r"n\'t", " not ", text)
	text = re.sub(r"i\'m", "i am ", text)
	text = re.sub(r"I\'m", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r"\'s"," is",text)
	text = re.sub(r"[0-9]"," ",str(text))
	words = text.split()

	return " ".join(word.lower() for word in words if word.lower() not in stopword)