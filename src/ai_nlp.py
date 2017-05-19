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
		return stemmer.stem(word)
	word_list = [reduce(word) for word in word_list if not word in stop_set]
	# return ' '.join(word_list)
	return word_list

# reviews_file = open("../data/review_Sport_and_Outdoors_5.json")
# reviews_file = open("../data/review_Sport_and_Outdoors_5.json")

counter = 0
all_words = Counter()
# useful_words = set()
for l in parse("../data/reviews_Sports_and_Outdoors_5.json.gz"):
	review_cleaned = clean_up_review(l['reviewText'])
	# print review_cleaned
	all_words.update(review_cleaned)
	# print l['helpful'][0],l['helpful'][1]
	# print l['overall']
	# print "______________________"
	counter += 1
	# if counter > 3:
	# 	break
	if counter % 1000 == 0:
		print 'at review # ', counter

print all_words
print "_________________________________________"
print all_words.most_common(20000)
print "We have " + str(len(all_words)) + " words"

with open('../data/individual_words.txt', 'wb') as word_file:
	pickle.dump(list(all_words), word_file)

