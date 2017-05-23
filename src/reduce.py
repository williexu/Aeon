import json
import gzip
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pickle

with open('../data/reviews.txt', 'rb') as word_file:
	data_list = pickle.load(word_file)

with open('../data/ratings.txt', 'rb') as word_file:
	ratings = pickle.load(word_file)
	
def clean_up_review(review_text):
	word_list = review_text.split()
	#remove stop words that dont have much meaning
	stop_set = set(stopwords.words('english'))
	word_list = [word for word in word_list if not word in stop_set]
	#can add porter stemming and lemmatizing
	stemmer = PorterStemmer()
	#lemmatizing
	lemmatizer = WordNetLemmatizer()
	def reduce(word):
		lemmatized = lemmatizer.lemmatize(word)
		return stemmer.stem(lemmatized)
	word_list = [reduce(word) for word in word_list]
	# return ' '.join(word_list)
	return word_list

clean_and_join = lambda review: ' '.join(clean_up_review(review))
data_list = map(clean_and_join, data_list)

with open('../data/reviews_cleaned.txt', 'wb') as word_file:
	pickle.dump(data_list, word_file)

print "finished cleaning"

