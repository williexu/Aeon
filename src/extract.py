import json
import gzip
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import pickle
import numpy
import cPickle

with open('../data/reviews_cleaned.txt', 'rb') as word_file:
	reviews = pickle.load(word_file)

# with open('../data/ratings.txt', 'rb') as word_file:
# 	ratings = pickle.load(word_file)

# with open('../data/helpfuls.txt', 'rb') as word_file:
# 	helpfuls = pickle.load(word_file)


for i in [100,200,500,1000,2000]:
	vectorizer = CountVectorizer(analyzer = "word",
	                             tokenizer = None, 
	                             preprocessor = None,
	                             stop_words = None,
	                             max_features = i) 

	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of 
	# strings.
	train_data_features = vectorizer.fit_transform(reviews)

	# Numpy arrays are easy to work with, so convert the result to an 
	# array
	train_data_features = train_data_features.toarray()

	print train_data_features.shape

	vocab = vectorizer.get_feature_names()
	# print vocab
	dist = numpy.sum(train_data_features, axis=0)

	# For each, print the vocabulary word and the number of times it 
	# appears in the training set
	for tag, count in zip(vocab, dist):
	    print count, tag

	print type(train_data_features)
	# print dist
	print type(dist)


	with open('../data/ratings.txt', 'rb') as file1:
		ratings = pickle.load(file1)

	with open('../data/helpfuls.txt', 'rb') as file2:
		helpfuls = pickle.load(file2)

	# features = numpy.array([numpy.concatenate((v,[1])) for v in train_data_features])
	features = numpy.array(train_data_features)
	
	labels = numpy.array(ratings)
	helpfuls = numpy.array(helpfuls)

	print features.shape
	print labels.shape
	print helpfuls.shape

	# randomly shuffle features and labels
	numpy.random.seed(28)
	numpy.random.shuffle(features)
	numpy.random.seed(28)
	numpy.random.shuffle(labels)
	numpy.random.seed(28)
	numpy.random.shuffle(helpfuls)	

	num = int(labels.shape[0])/20

	x = features[:num]
	y = labels[:num]
	z = helpfuls[:num]

	x = numpy.array([numpy.concatenate((v,[1])) for v in x])

	print x.shape
	print y.shape
	print z.shape

	with open('../data/review_features_sample'+ str(i)+ '.txt', 'wb') as word_file:
		pickle.dump(x, word_file)

	with open('../data/ratings_sample.txt', 'wb') as word_file:
		pickle.dump(y, word_file)

	with open('../data/helpfuls_sample.txt', 'wb') as word_file:
		pickle.dump(z, word_file)

	# p = cPickle.Pickler(open("../data/review_features.txt","wb")) 
	# p.fast = True 
	# p.dump(train_data_features) 

	with open('../data/vocab'+ str(i)+'.txt', 'wb') as word_file:
		pickle.dump(vocab, word_file)

