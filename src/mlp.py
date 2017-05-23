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
valid_len = 1000

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

def create_mlp(in_dim=1001, layers=1, nodes=[]):
	model=Sequential()
	model.add(Dense(units=nodes[0],input_dim=in_dim))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	for layer in range(layers):	
		print "adding layer"
		model.add(Dense(units=nodes[layer+1]))
		model.add(Activation('relu'))
		model.add(Dropout(0.2))
	model.add(Dense(units=1))
	#model.add(Activation('sigmoid'))
	model.compile(optimizer='rmsprop',loss='mse', metrics=['mae'])
	model.fit(x1,y1,epochs=50)
	return model

def correct_preds(preds):
	for i in range(len(preds)):
		if preds[i] < 0:
			preds[i]==0
		if preds[i] > 5:
			preds[i]==5
#		if preds[i]*10%10 < 5:
#			preds[i]=int(preds[i])*1.0
#		else:
#			preds[i]=(int(preds[i])+1.0)*1.0	
	return preds
res=[]

#for layers in range(5):
nodes=[100,5,4,3,2]
model = create_mlp(in_dim=2001,layers = 4,nodes=nodes)
preds = model.predict(xValid,batch_size=valid_len)
preds=correct_preds(preds)
print
print"Mean squared error: "
print sum((preds-yValid)**2)/len(yValid)
#res.append([nodes[1:layers+1], float(sum((preds-yValid)**2)/len(yValid))])

model.save('mlp-model.h5')
# for layers in range(5):
# 	nodes=[10,5,4,3,2]
# 	model = create_mlp(in_dim=1001,layers = layers,nodes=[20,10,5,4,3,2])
# 	preds = model.predict(xValid,batch_size=valid_len)
# 	preds=correct_preds(preds)
# 	print
# 	print"Mean squared error: "
# 	print sum((preds-yValid)**2)/len(yValid)
# 	res.append([nodes[1:layers+1],float(sum((preds-yValid)**2)/len(yValid))])
	
print len(res)
print	
for i in res:
	print i	
#with open('../data/mlp_results-1000.txt', 'wb') as word_file:
#	pickle.dump(res, word_file)
