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


def parse(path):
	g = gzip.open(path, 'r')
	for l in g:
		yield eval(l)

def clean_up_review(review_text):
	#remove html
	pretty = BeautifulSoup(review_text).get_text()
	letters = re.sub('[^a-zA-Z]', ' ', pretty)
	word_list = letters.lower().split()
	#remove stop words that dont have much meaning
	stop_set = set(stopwords.words('english'))
	word_list = [word for word in word_list if not word in stop_set]
	#can add porter stemming and lemmatizing
	stemmer = PorterStemmer()
	#lemmatizing
	lemmatizer = WordNetLemmatizer()
	def reduce(word):
		# stemmed = stemmer.stem(word)
		# return lemmatizer.lemmatize(stemmed)
		return lemmatizer.lemmatize(word)
		# return stemmer.stem(word)
	word_list = [reduce(word) for word in word_list if not word in stop_set]
	# return ' '.join(word_list)
	return word_list

counter = 0
all_words = Counter()
# useful_words = set()
for l in parse("../data/reviews_Sports_and_Outdoors_5.json.gz"):
	review_cleaned = clean_up_review(l['reviewText'])
	# print review_cleaned
	all_words.update(review_cleaned)
	counter += 1
	if counter % 1000 == 0:
		print 'at review # ', counter

print all_words
print "_________________________________________"
print all_words.most_common(20000)
print "We have " + str(len(all_words)) + " words"

with open('../data/individual_words_lemmatized.txt', 'wb') as word_file:
	pickle.dump(all_words, word_file)
