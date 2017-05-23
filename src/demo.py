import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from ai_spell_checker import correct
import math

features = 2000

with open('../data/ridge'+str(features)+'.txt', 'rb') as file1:
    	model = pickle.load(file1) # use own word file

with open('../data/vocab'+str(features)+'.txt', 'rb') as file2:
    	vocab = pickle.load(file2) # use own word file

stop_set = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_up_review(review_text):
	letters = re.sub('[^a-zA-Z]', ' ', review_text)
	word_list = letters.lower().split()
	#lemmatizing
	lemmatizer = WordNetLemmatizer()
	def reduce(word):
		return lemmatizer.lemmatize(correct(word))
	word_list = [reduce(word) for word in word_list]
	return word_list

def reduce(word):
	lemmatized = lemmatizer.lemmatize(word)
	return stemmer.stem(lemmatized)

bound = lambda a : max(min(a,5),1)
# bound = lambda a : a
bound = numpy.vectorize(bound)

vector = [0] * len(vocab)

review = raw_input("Enter a review: \n")

while len(review) <= 10:
	review = raw_input("Please enter a longer review: \n")

print type(model)

#average length of a review is ~600 so we scale
scale = 600.0 / len(review)
scale = int(math.ceil(scale))

word_list = clean_up_review(review)
# print word_list[:min(len(word_list), 30)]
# print 'length', len(word_list)
word_list = [word for word in word_list if not word in stop_set]
# print word_list[:min(len(word_list), 30)]
# print 'length',len(word_list)
word_list = [reduce(word) for word in word_list]
# print word_list[:min(len(word_list), 30)]

# print type(vocab)
# print len(vocab)
# print vocab

for x in word_list:
	if x in vocab:
		vector[vocab.index(x)] += 1 * scale
		print 'match', x

vector = numpy.array([numpy.concatenate((vector,[1]))])
print vector.shape

print bound(model.predict(vector))

