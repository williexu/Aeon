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
from ai_spell_checker import *


def parse(path):
	g = gzip.open(path, 'r')
	for l in g:
		yield eval(l)

def clean_up_review(review_text):
	letters = re.sub('[^a-zA-Z]', ' ', review_text)
	word_list = letters.lower().split()
	#can add porter stemming and lemmatizing
	# stemmer = PorterStemmer()
	#lemmatizing
	lemmatizer = WordNetLemmatizer()
	def reduce(word):
		# stemmed = stemmer.stem(word)
		# return lemmatizer.lemmatize(stemmed)
		return lemmatizer.lemmatize(correct(word))
		# return stemmer.stem(word)
	word_list = [reduce(word) for word in word_list]
	# return ' '.join(word_list)
	return word_list

counter = 0
reviews = ['somestring'] * 296337
ratings = [1] * 296337
helpfuls = [None] * 296337

# useful_words = set()
for l in parse("../data/reviews_Sports_and_Outdoors_5.json.gz"):
	# review_cleaned = clean_up_review(l['reviewText'])
	# reviews[counter] = ' '.join(review_cleaned)
	# ratings[counter] = l['overall']
	helpfuls[counter] = l['helpful']
	

	counter += 1
	if counter % 10 == 0:
		print 'at review # ', counter
# print counter

# with open('../data/reviews.txt', 'wb') as word_file:
# 	pickle.dump(reviews, word_file)

# with open('../data/ratings.txt', 'wb') as word_file:
# 	pickle.dump(ratings, word_file)

with open('../data/helpfuls.txt', 'wb') as word_file:
	pickle.dump(ratings, word_file)



