import csv
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
