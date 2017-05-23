import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.layers import Embedding,LSTM
import pickle
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler

with open('../data/review_features_sample2000.txt','r') as f:
	x1 = pickle.load(f)

with open('../data/ratings_sample.txt','r') as f:
	y1 = pickle.load(f)

with open('../data/helpfuls_sample.txt','r') as f:
	y2 = pickle.load(f)



y1=y1.reshape(14816,1)
sep=np.arange(len(y1))
np.random.shuffle(sep)
valid_len = 500

xValid=x1[sep[:valid_len]]
yValid=y1[sep[:valid_len]]
x1=x1[sep[valid_len:]]
y1=y1[sep[valid_len:]]

scaler = StandardScaler().fit(x1)
x1=scaler.transform(x1)
xValid=scaler.transform(xValid)

print x1.shape
print y1.shape
print xValid.shape
print yValid.shape

def create_recurr():
	model=Sequential()
	model.add(Embedding(2000,10))
	model.add(LSTM(10,dropout=0.2,recurrent_dropout=0.2))
	model.add(Dense(units=1))
	model.compile(optimizer='rmsprop',loss='mse', metrics=['mae'])

	model.fit(x1,y1,epochs=3)
	return model

model = create_recurr()
score = model.evaluate(xValid,yValid)
preds = model.predict(xValid)

print
print"Mean squared error: "
tmp=0
for i in range(len(preds)):
	tmp+= (preds[i]-yValid[i]) **2
print tmp*1.0/len(yValid)
print sum((preds-yValid)**2)/len(yValid)
#print " " 