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
# import pylab as pl

for i in [100,200,500,1000,2000]:
    with open('../data/review_features_sample'+str(i)+'.txt', 'rb') as file1:
    	features = pickle.load(file1) # use own word file

    with open('../data/ratings_sample.txt', 'rb') as file2:
    	labels = pickle.load(file2)

    with open('../data/vocab'+str(i)+'.txt', 'rb') as file3:
    	vocab = pickle.load(file3)


    print features.shape
    print labels.shape


    print "doing ordinary least squares"

    # Create linear regression object
    linreg = LinearRegression(normalize = True)

    # # randomly shuffle features and labels
    numpy.random.seed(90)
    numpy.random.shuffle(features)
    numpy.random.seed(90)
    numpy.random.shuffle(labels)

    # x = features[:5000]
    # y = labels[:5000]
    # test = features[5000:]
    # test_labels = labels[5000:]

    x = features
    y = labels

    # Train the model using the training sets
    linreg.fit(x,y)

    # preds = linreg.predict(test[:1000])
    # actual =  test_labels[:1000]

    bound = lambda a : max(min(a,5),1)
    # bound = lambda a : a
    bound = numpy.vectorize(bound)

    # print preds
    # print bound(preds)
    # print actual
    # print preds - actual
    # print abs(preds - actual)

    # print sum(abs(bound(preds) - actual)**2)/1000.
    # rand_labels = numpy.copy(test_labels)
    # numpy.random.shuffle(rand_labels)
    # print sum(abs(test_labels-rand_labels)**2)/float(len(test_labels))
    # print type(linreg)


    # print train.shape
    # print train_labels.shape
    # print test.shape
    # print test_labels.shape

    # Compute MSE on training data
    # p = np.array([linreg.predict(xi) for xi in x])
    p = linreg.predict(x)
    # Now we can constuct a vector of errors
    err = abs(bound(p)-y)

    # Let's see the error on the first 10 predictions
    print err[:10]

    # Dot product of error vector with itself gives us the sum of squared errors
    total_error = numpy.dot(err,err)
    # Compute MSE
    mse_train = total_error/len(p)
    print mse_train

    # We can view the regression coefficients
    print 'Regression Coefficients: \n', linreg.coef_

    indices_sorted = numpy.argsort(linreg.coef_)
    print "Low sentiment vocab", [vocab[i] for i in indices_sorted[:5]]
    print "High sentiment vocab", [vocab[i] for i in indices_sorted[-5:]]


    # # Plot outputs
    # # %matplotlib inline
    # pl.plot(p, y,'ro')
    # pl.plot([0,10],[0,10], 'g-')
    # pl.xlabel('predicted')
    # pl.ylabel('real')
    # pl.show()


    # Now let's compute MSE using 10-fold x-validation
    kf = KFold(10)
    xval_err = 0
    for train,test in kf.split(x):
        linreg.fit(x[train],y[train])
        # p = np.array([linreg.predict(xi) for xi in x[test]])
        p = linreg.predict(x[test])
        e = bound(p)-y[test]
        xval_err += numpy.dot(e,e)
        
    mse_10cv = xval_err/len(x)

    method_name = 'Simple Linear Regression'
    print('Method: %s' %method_name)
    print('MSE on training: %.4f' %mse_train)
    print('MSE on 10-fold CV: %.4f' %mse_10cv)


    # Create linear regression object with a ridge coefficient 0.5
    ridge = Ridge(fit_intercept=True, alpha=0.5, normalize = True)

    # Train the model using the training set
    ridge.fit(x,y)

    # Compute RMSE on training data
    # p = np.array([ridge.predict(xi) for xi in x])
    p = ridge.predict(x)
    err = bound(p)-y
    total_error = numpy.dot(err,err)
    mse_train = total_error/len(p)

    # Compute MSE using 10-fold x-validation
    kf = KFold(10)
    xval_err = 0
    for train,test in kf.split(x):
        ridge.fit(x[train],y[train])
        p = ridge.predict(x[test])
        e = bound(p)-y[test]
        xval_err += numpy.dot(e,e)
    mse_10cv = xval_err/len(x)

    method_name = 'Ridge Regression'
    print('Method: %s' %method_name)
    print('RMSE on training: %.4f' %mse_train)
    print('RMSE on 10-fold CV: %.4f' %mse_10cv)


    print('Ridge Regression')
    print('alpha\t MSE_train\t MSE_10cv\n')
    alpha = numpy.linspace(0,5,20)
    t_mse = numpy.array([])
    cv_mse = numpy.array([])

    for a in alpha:
        ridge = Ridge(fit_intercept=True, alpha=a, normalize = True)
        
        # computing the RMSE on training data
        ridge.fit(x,y)
        p = ridge.predict(x)
        err = bound(p)-y
        total_error = numpy.dot(err,err)
        mse_train = total_error/len(p)

        # computing RMSE using 10-fold cross validation
        kf = KFold(10)
        xval_err = 0
        for train, test in kf.split(x):
            ridge.fit(x[train], y[train])
            p = ridge.predict(x[test])
            err = bound(p) - y[test]
            xval_err += numpy.dot(err,err)
        mse_10cv = xval_err/len(x)
        
        t_mse = numpy.append(t_mse, [mse_train])
        cv_mse = numpy.append(cv_mse, [mse_10cv])
        print('{:.3f}\t {:.4f}\t\t {:.4f}'.format(a,mse_train,mse_10cv))


    # pl.plot(alpha, t_mse, label='MSE_train')
    # pl.plot(alpha, cv_mse, label='MSE_CrossVal')
    # pl.legend( ('MSE_train', 'MSE_CrossVal') )
    # pl.ylabel('MSE')
    # pl.xlabel('alpha')
    # pl.show()


    a = 0.3
    for name,met in [
            ('linear regression', LinearRegression()),
            ('lasso', Lasso(fit_intercept=True, alpha=a, normalize = True)),
            ('ridge', Ridge(fit_intercept=True, alpha=a, normalize = True)),
            ('elastic-net', ElasticNet(fit_intercept=True, alpha=a, normalize = True))
            ]:
        met.fit(x,y)
        # p = np.array([met.predict(xi) for xi in x])
        p = met.predict(x)
        e = bound(p)-y
        total_error = numpy.dot(e,e)
        mse_train = total_error/len(p)

        kf = KFold(10)
        err = 0
        for train,test in kf.split(x):
            met.fit(x[train],y[train])
            p = met.predict(x[test])
            e = bound(p)-y[test]
            err += numpy.dot(e,e)

        mse_10cv = err/len(x)
        print('Method: %s' %name)
        print('MSE on training: %.4f' %mse_train)
        print('MSE on 10-fold CV: %.4f' %mse_10cv)
        print "\n"
