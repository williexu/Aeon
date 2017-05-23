import numpy as np
#import theano
import csv
from sklearn import neighbors, tree,decomposition
from sklearn import svm
#from statsmodels.tsa.arima_model import ARIMA
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
from sklearn import ensemble
#import statsmodels.api as sm
#from statsmodels.graphics.api import qqplot
#import matplotlib.patches as mpatches
#import pylab
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,Embedding,LSTM
#from theano import ifelse
def one_hot(line):
	for i,x in enumerate(line):
		if float(x)==1.0:
			return i
x=[]
y=[]
xTr=[]
yTr=[]
xTe=[]


with open('X.csv','r') as f:
	reader=csv.reader(f,delimiter=",")
	for line in reader:
		#print np.asarray(line)
		xTr.append(np.asarray(line))
		x.append(np.asarray(line))

with open('Y.csv','r') as f:
	reader=csv.reader(f,delimiter=",")
	for line in reader:
		#num=one_hot(line)
		yTr.append(line)
		y.append(line)

with open('testX.csv','r') as f:
	reader=csv.reader(f,delimiter=",")
	for line in reader:
		xTe.append(line)		


xTr=np.asarray(xTr).astype(np.float)
yTr=np.asarray(yTr).astype(np.float)#.reshape(-1,1)
xTe=np.asarray(xTe).astype(np.float)
x=np.asarray(x).astype(np.float)
y=np.array(y).astype(np.float)

# pca_ = decomposition.PCA(n_components=1)
# x_ = pca_.fit_transform(x).reshape(1,177)
# print x_
# tmp=np.argsort(x_[0]).reshape(1,177)
# print tmp
# y=y.reshape(177,1)
# plt.xlabel("Principal Component ")
# plt.ylabel("Penicillin")
# plt.plot(y[tmp][0])
# plt.show()

#pca_=decomposition.PCA(n_components=10)
#xTr=pca_.fit_transform(xTr)
#x=pca_.fit_transform(x)
#x=np.ones((177,110))
# model=ARIMA(x[0][3000:3600],order=(10,1,1))
# print model.fit(disp=0).params
# x_arma=[]
# for i in range(177):
# 	x_i=[]
# 	for b,j in enumerate([0,600,1200,1800,2400,3000,3600,4200,4800,5400]):
# 		print i, j
# 		if j==3000:
# 			print "in here"
# 			model=ARIMA(x[i][j:j+600],order=(10,1,1))
# 			x_i.append(model.fit(disp=0).params)
# 			continue
# 		model=ARIMA(x[i][j:j+600],order=(10,1,0))
# 		x_i.append(model.fit(disp=0).params)
# 	x_i=np.asarray(x_i).flatten()
# 	x_arma.append(x_i)
# x=np.asarray(x_arma)
# #		for z in len(model_fit.params):
# #			x[i][b*11+z]=model_fit.params[z]
# print x
#plt.plot(x[1][1200:1800])
#plt.show()

# m1=Sequential()
# m2=Sequential()
# m3=Sequential()
# m4=Sequential()
# m5=Sequential()
# m6=Sequential()
# m7=Sequential()
# m8=Sequential()
# m9=Sequential()
# m10=Sequential()
# models=[m1,m2,m3,m4,m5,m6,m7,m8,m9,m10]
# x1=x[:][0:600].T
# x2=x[:][600:1200].T
# x3=x[:][1200:1800].T
# x4=x[:][1800:2400]
# x5=x[:][2400:3000].T
# x6=x[:][3000:3600].T
# x7=x[:][3600:4200].T
# x8=x[:][4200:4800].T
# x9=x[:][4800:5400].T
# x10=x[:][5400:6000].T
# data=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]

# for i,models in enumerate(models):
# 	models.add(Dense(units=100,input_dim=600)) #614656 for fully interconnected
# 	models.add(Activation('relu'))
# 	models.add(Dropout(0.2))
# 	models.add(Dense(units=1))
# 	models.add(Activation('softmax'))
# 	models.compile([optimizer='rmsprop',loss='mse',metrics='mae'])
# 	models.fit(data[i],epochs=15)

# m = Sequential()
# 	m.add(Dense(units=10,input_dim=10))
# 	m.add(Activation('relu'))
# 	m.add(Dropout(0.2))
# 	m.add(Dense(units=1))
# 	m.add(Activation('softmax'))
# 	m.compile([optimizer='rmsprop',loss='mse',metrics='mae'])
model3=Sequential()
model3.add(Embedding(20000,128))
model3.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
model3.add(Dense(1,activation='sigmoid'))
model3.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
model3.fit(xTr,yTr,epochs=20)
score=model3.evaluate(xTe,yTe)

print(score)
sep=np.arange(len(yTr))
np.random.shuffle(sep)
xValid=xTr[sep[:20]]
yValid=yTr[sep[:20]]
xTr=xTr[sep[20:]]
yTr=yTr[sep[20:]]

print(x.shape)
print(y.shape)
print(xValid.shape)

knn_eps = []

for i in range(100):   #KNN with size 2 with smallest loss function, 0.001047
	neigh = neighbors.KNeighborsRegressor(n_neighbors=i+1)
	neigh.fit(xTr, yTr) 
	preds = neigh.predict(xValid)
	#print [preds]
	knn_eps.append(0.05*sum(preds-yValid)**2)
	print("KNN with " + str(i)+":", 0.05*sum(preds-yValid)**2)
#	if .05*sum(preds-yValid)**2 < 0.00005:
#		break



plt.xlabel("Number of Nearest Neighbors")
plt.ylabel("Mean Squared Error")
plt.plot(knn_eps,'b')
#plt.show()


# #print x_.explained_variance
# x_ = x_.reshape(1,20)
# tmp=np.argsort(x_)
# #print np.sort(y.reshape(1,177))
# print tmp.shape
# red_patch = mpatches.Patch(color='red', label='The predicted results')
# blue_patch = mpatches.Patch(color='blue', label='The real results')
# plt.legend(handles=[red_patch,blue_patch])
# plt.xlabel('First Principal Component')
# plt.ylabel('Penicillin Concentration')
# plt.plot(yValid[tmp][0],'b',label="Actual Results")
# plt.plot(preds[tmp][0],'r',label="Predicted Results")
# plt.show()

d_tree=[]
for i in range(1,100): #Single Decision Tree with smallest loss funciton, depth 15-16 with  loss function 0.00006, 0.00008
	clf = tree.DecisionTreeRegressor(max_depth=i)
	clf=clf.fit(xTr,yTr)
	preds = clf.predict(xValid)
	preds=preds.reshape(-1,1)
	print("Decision Tree Regression with " + str(i)+ ": ", 0.05*sum(preds-yValid)**2)
	d_tree.append(0.05*sum(preds-yValid)**2)

plt.xlabel("Depth of Tree")
plt.ylabel("Mean Squared Error")
plt.plot(d_tree,'r')
#plt.show()

b_tree=[]
res=[]
for i in range(30):
	r = np.random.randint(0,xTr.shape[0],size=xTr.shape[0])
	clf = tree.DecisionTreeRegressor(max_depth=50)
	clf=clf.fit(xTr[r],yTr[r])
	preds = clf.predict(x)
	#preds=preds.reshape(-1,1)
	res.append(preds)

res = sum(np.asarray(res))/30
print("Bagging Decision Tree with 50 different trees: ", .05*sum(res.reshape(-1,1)-y)**2)

# # neigh = neighbors.KNeighborsRegressor(n_neighbors=4)
# # neigh.fit(xTr, yTr) 
# # preds = neigh.predict(xValid)
# # #plt.plot(preds,'r')
# # #plt.plot(yValid,'b')
# # #plt.show()
# # #plt.savefig('Bagging_Tree.png')
random_forest_eps=[]
for i in range(100):
	i
	f = ensemble.RandomForestRegressor(n_estimators=i+1,max_features=2)
	f.fit(xTr,yTr)
	preds = f.predict(xValid)
	preds=preds.reshape(-1,1)
	random_forest_eps.append(0.5*sum(preds-yValid)**2)
	print("Random Forest with " + str(i) + "trees: ", .05*sum(preds-yValid)**2)
	# if .05*sum(preds-y)**2 < 0.00005:
	# 	break


red_patch = mpatches.Patch(color='red', label='Decision Tree Errors')
blue_patch = mpatches.Patch(color='blue', label='KNN errors')
green_patch = mpatches.Patch(color='green',label='Random Forest Errors')
plt.legend(handles=[red_patch,blue_patch,green_patch])

plt.xlabel("Number of Parameters")
plt.ylabel("Mean Squared Error")
plt.plot(random_forest_eps,'g')
plt.show()

#for i in range(10):
#	clf=svm.SVC(decision_function_shape='ovo',C=i+1)
#	clf.fit(xTr,yTr)
#	dec = clf.decision_function([[1]])
#	clf.decision_function_shape = "ovr"
#	dec = clf.decision_function([[1]])
#	dec.shape[1] # 4 classes
