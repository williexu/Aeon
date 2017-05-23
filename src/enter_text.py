import numpy as np
import csv
from keras.models import load_model
#from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout
#from keras.layers import Embedding,LSTM
import pickle
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
import argparse
import sys
import pickle
from ai_spell_checker import *
import numpy as np

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

if __name__ == "__main__":
	with open('../data/vocab'+sys.argv[1]+'.txt','r') as f:
		vocab = pickle.load(f)
	with open('../data/review_features_sample'+sys.argv[1]+'.txt','r') as f:
		x=pickle.load(f)

	test_string = raw_input("Please type your hypothetical review here: \n")
	test_string=clean_up_review(test_string)
	print test_string
	test_x = np.zeros(len(vocab)+1)
	for word in test_string:
		for i in range(len(vocab)):
			if word == vocab[i]:
				test_x[i]+=1
	
	print sum(test_x)
	test_x=test_x.reshape(1,2001)
	#print sum(test_x)
	print
	print
	scaler = StandardScaler().fit(test_x)
	test_x=scaler.transform(test_x)
 
	model = load_model('mlp-model.h5')
	preds = model.predict(test_x)
	print preds