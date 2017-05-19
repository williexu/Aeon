import json
import string
from pprint import pprint
from theano import ifelse
import nltk
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation
import pickle


#our code goes here
y1=[]
y2=[]
data={}
with open('../data/Sports_and_Outdoors_5.json','r') as f:
        with open('../data/words_spell_checked.txt','r') as f1:
                dictionary=pickle.load(f1)
	for line in f:
		tmp = json.loads(line)
		l=str(tmp["reviewText"])
		overall=int(tmp["overall"])
		y1.append(overall)
		y2.append( ((float)(tmp["helpful"][0]),( (float)(tmp["helpful"][1]))) )
		#print l
		l = nltk.word_tokenize(l)
		for i in l:
#			print i
			i=i.translate(None,string.punctuation)
			try:
				data[i]+=1
			except:
				data[i]=1


print y1[0]
print y2[0]
print data.keys()
print len(data.keys())

# model=Sequential()
# model.add(Dense(units=64,input_dim=len(data.keys())))
# model.add(Activation('sigmoid'))
# model.add(Dense(units=64))
# model.add(Activation('sigmoid'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])

# model.fit(x,y,batches=128

#import gzip
#from bs4 import BeautifulSoup
#import re
#import nltk
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#from nltk.stem import WordNetLemmatizer
#from collections import Counter
#import pickle
#from ai_spell_checker import *


# def parse(path):
# 	g = gzip.open(path, 'r')
# 	for l in g:
# 		yield eval(l)

# def clean_up_review(review_text):
# 	letters = re.sub('[^a-zA-Z]', ' ', review_text)
# 	word_list = letters.lower().split()
# 	#can add porter stemming and lemmatizing
# 	# stemmer = PorterStemmer()
# 	#lemmatizing
# 	lemmatizer = WordNetLemmatizer()
# 	def reduce(word):
# 		# stemmed = stemmer.stem(word)
# 		# return lemmatizer.lemmatize(stemmed)
# 		return lemmatizer.lemmatize(correct(word))
# 		# return stemmer.stem(word)
# 	word_list = [reduce(word) for word in word_list]
# 	# return ' '.join(word_list)
# 	return word_list

# counter = 0
# reviews = []
# ratings = []

# # useful_words = set()
# for l in parse("../data/reviews_Sports_and_Outdoors_5.json.gz"):
# 	review_cleaned = clean_up_review(l['reviewText'])
# 	reviews.append(' '.join(review_cleaned))
# 	ratings.append(l['overall'])
	

# 	counter += 1
# 	if counter % 10 == 0:
# 		print 'at review # ', counter

# with open('../data/reviews.txt', 'wb') as word_file:
# 	pickle.dump(reviews, word_file)

# with open('../data/ratings.txt', 'wb') as word_file:
# 	pickle.dump(ratings, word_file)

