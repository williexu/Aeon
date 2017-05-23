import gzip
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pickle
import numpy
from run import *

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)
# def clean_up_review(review_text):
#     letters = re.sub('[^a-zA-Z]', ' ', review_text)
#     letters=review_text
#     word_list = letters.lower().split()
#     #can add porter stemming and lemmatizing
#     # stemmer = PorterStemmer()
#     #lemmatizing
#     lemmatizer = WordNetLemmatizer()
#     def reduce(word):
#         # stemmed = stemmer.stem(word)
#         # return lemmatizer.lemmatize(stemmed)
#         return lemmatizer.lemmatize(correct(word))
#         # return stemmer.stem(word)
#     word_list = [reduce(word) for word in word_list]
#     # return ' '.join(word_list)
#     return word_list

counter = 0
reviews = []
ratings = []
helpful = []
# useful_words = set()
for l in parse("../data/reviews_Sports_and_Outdoors_5.json.gz"):
    txt=l['reviewText'].lower()
    review = nltk.sent_tokenize(txt)
    review = map(lambda x: clean_up_review(x),review)
    print
    print review
    if counter==10:
        break
    reviews.append(review)
    ratings.append(l['overall'])
    helpful=[l['helpful'][0],l['helpful'][1]]
    counter += 1
    if counter % 10 == 0:
        print 'at review # ', counter

with open('../data/reviews_for_recurrent.txt', 'wb') as word_file:
    pickle.dump(reviews, word_file)

with open('../data/ratings.txt', 'wb') as word_file:
    pickle.dump(ratings, word_file)

with open('../data/helpful.txt','wb') as word_file:
    pickle.dump(helpful, word_file)