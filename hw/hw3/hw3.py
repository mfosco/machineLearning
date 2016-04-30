# Michael Fosco, mfosco
# 4/12/16

import numpy as np 
import pandas as pd 
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
import multiprocessing as mp
from multiprocessing import Process, Queue
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import os, timeit, sys, itertools, re
import seaborn as sns
from sklearn import linear_model, neighbors, ensemble, svm, preprocessing
from numba.decorators import jit, autojit
from numba import double #(nopython = True, cache = True, nogil = True [to run concurrently])


'''
example multiprocess:
def f(i, y, X, lamSims, tN, thetaN, epsHat):
    return simulateOne(y, X, lamSims, tN, thetaN, epsHat)


def boot(y, X, lamSims, num):
	res = thetaHatN(X,y)
	thetaN = res[0]
	tN = abs(res[1])
	dat = X[:,res[2]]
	dat = sm.add_constant(dat, prepend = False)
	t = sm.OLS(y, dat).fit()
	epsHat = t.resid
	simul = [0]*num

	for j in xrange(num):
		simul[j] = simulateOne(y, X, lamSims, tN, thetaN, epsHat)

	return simul
'''

'''
Notes from hw2: get rid of special cases for plots
just make one plot per param then decide how to arrange them in the report

delete densityPlots
probs split pipeline into further pieces like:
fit model
then get desire statistic for goodness of fit

need to impute test set using train set statistics
'''

'''
Note: model.predict( x) predicts from model using x
'''
##################################################################################
'''
List of models and parameters
'''

cores = 4
modelLR = {'model': LogisticRegression, 'solver': ['liblinear'], 'C' : [.01, .1, .5, 1, 5, 10, 25],
		  'class_weight': ['balanced', None], 'n_jobs' : [cores],
		  'tol' : [1e-7, 1e-5, 1e-4, 1e-3, 1e-1, 1]}
modelLSVC = {'model': svm.LinearSVC, 'tol' : [1e-7, 1e-5, 1e-4, 1e-3, 1e-1, 1], 'class_weight': ['balanced', None],
			 'max_iter': [1000, 2000], 'C' :[.01, .1, .5, 1, 5, 10, 25]}
modelKNN = {'model': neighbors.KNeighborsClassifier, 'weights': ['uniform', 'distance'], 'n_neighbors' : [2, 5, 10, 50, 100, 500, 1000],
			'leaf_size': [15, 30, 60, 120], 'n_jobs': [cores]}



modelList = [modelLR, modelLSVC, modelKNN]
#X_scaled = preprocessing.scale(X) gets scaled version of X

##################################################################################

'''
Read in the data functions
'''

'''
converts from camel case to snake case
Taken from:  http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
'''
def camel_to_snake(column_name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

'''
Read data from a csv
'''
def readcsv(filename):
	assert(type(filename) == str and filename.endswith('.csv'))
	data = pd.read_csv(filename, index_col = 0)
	#data.columns = [camel_to_snake(col) for col in data.columns]
	return data

###############################################################
'''
Descriptive functions
'''

'''
Create a correlation table
'''
def corrTable(df, method = 'pearson', min_periods = 1):
	return df.corr(method, min_periods)

'''
Generate a descriptive Table
'''
def descrTable(df):
	sumStats = df.describe(include = 'all')
	missingVals = len(df.index) - df.count()

	oDF = pd.DataFrame(index = ['missing values'], columns = df.columns)

	for col in df.columns:
		oDF.ix['missing values',col] = missingVals[col]

	fDF = sumStats.append(oDF)
	return fDF

'''
Make bar plots
'''
def barPlots(df, items, saveExt = ''):
	for it in items:
		b = df[it].value_counts().plot(kind = 'bar', title = it)
		s = saveExt + it + '.pdf'
		b.get_figure().savefig(s)
		plt.show()

'''
Make pie plots.
'''
def piePlots(df, items, saveExt = ''):
	for it in items:
		b = df[it].value_counts().plot(kind = 'pie', title = it)
		s = saveExt + it + 'Bar.pdf'
		b.get_figure().savefig(s)
		plt.show()

'''
Discretize a continous variable. 
num: 		The number of buckets to split the cts variable into
'''
def discretize(df, cols, num=10):
	dDF = df
	for col in cols:
		dDF[col] = pd.cut(dDF[col], num)
	return dDF

'''
Convert categorical variables into binary variables
'''
def categToBin(df, cols):
	dfN = df
	for col in cols:
		dfN = pd.get_dummies(df[col])
	df_n = pd.concat([df, dfN], axis=1)
	return df_n

'''
Helper function to make histograms
'''
def makeHisty(ax, col, it, binns = 20):
	n, bins, patches = ax.hist(col, binns=20, histtype='bar', range=(min(col), max(col)))

'''
Make histogram plots, num is for layout of plot
'''
def histPlots(df, items, fname, binns = 20, saveExt = ''):
	indx = 1

	num = len(items)
	iters = num % 4
	z = 0

	for i in range(0, iters):
		fig, axarr = plt.subplots(2, 2)
		x = 0
		y = 0

		for it in items[z:z+4]:
			makeHisty(axarr[x,y], df[it], it, binns)
			axarr[x,y].set_title(it)
			y += 1
			if y >= len(axarr):
				x += 1
				y = 0
			if x >= len(axarr):
				break
		fig.savefig(saveExt + fname + str(indx) + 'Hists.pdf')
		plt.clf()
		indx += 1
		z += 4

	leftover = num - z
	leftIts = items[z:]

	if leftover == 1:
		fig, axarr = plt.subplots(1,1)
		makeHisty(axarr[0], df[leftIts[0]], leftIts[0], binns)
		axarr[0].set_title(leftIts[0])
		fig.savefig(saveExt + fname + str(indx) + 'Hists.pdf')
		plt.clf()
	elif leftover == 2:
		fig, axarr = plt.subplots(1,2)
		x = 0
		for it in leftIts:
			makeHisty(axarr[x], df[it], it, binns)
			axarr[x].set_title(it)
			x += 1
		fig.savefig(saveExt + fname + str(indx) + 'Hists.pdf')
		plt.clf()	
	elif leftover == 3:
		fig, axarr = plt.subplots(1,3)
		x = 0
		for it in leftIts:
			makeHisty(axarr[x], df[it], it, binns)
			axarr[x].set_title(it)
			x += 1
		fig.savefig(saveExt + fname + str(indx) + 'Hists.pdf')
		plt.clf()	
	return


'''
Generate a density plot for certain items
'''
def densityPlots(df, items, saveExt = ''):
	for it in items:
		b = df[it].value_counts().plot(kind = 'kde', title = it + ' Density Plot')
		s = saveExt + it + 'DensityPlot.pdf'
		b.get_figure().savefig(s)
		plt.show()	

'''
takes a response series and a matrix of features, and uses a random
forest to rank the relative importance of the features for predicting
the response.    
Basically taken from the DataGotham2013 GitHub repo and scikit learn docs
'''   
def identify_important_features(X,y,save_toggle=False):
    forest = ensemble.RandomForestClassifier()
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    sorted_indices = np.argsort(importances)[::-1]

    padding = np.arange(len(X.columns)) + 0.5
    plt.barh(padding, importances[sorted_indices],color='r', align='center',xerr=std[sorted_indices])
    plt.yticks(padding, X.columns[sorted_indices])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    if save_toggle:
        plt.savefig('RFimportant_features.png')
    plt.show()
    
'''
Plot x vs y for each x in X
'''    
def x_vs_y_plots(X,y,save_toggle=False):
    df = pd.concat([X, pd.DataFrame(y, index=X.index)], axis=1)
    for x in X.columns:
        df[[x,y.name]].groupby(x).mean().plot()
        if save_toggle:
            plt.savefig(x+'_vs_'+y.name+'.png')
        plt.show()
    return

#################################################################################
'''
Fill in Data functions:
'''

'''
Fill in the mean for select items
'''
def fillNaMean(df):
	tdf = df

	numFields = tdf.select_dtypes([np.number])
	catFields = tdf.select_dtypes(['object','category'])

	if len(catFields.columns) > 0:
		for col in catFields.columns:
			ind = pd.isnull(tdf[col])
			fill_val = tdf[col].mode()[0]
			tdf.ix[ind,col] = fill_val

	if len(numFields.columns) > 0:
		for col in numFields.columns:
			ind = pd.isnull(tdf[col])
			fill_val = tdf[col].median()
			tdf.ix[ind,col] = fill_val
	return tdf

'''
Fill in the conditional mean for select items
'''
def fillConditMean(df, conditions):
	if type(conditions) != list:
		raise TypeError('Error in fillConditMean. Variable CONDITIONS was not of type list')

	dat = df

	numFields = dat.select_dtypes([np.number])
	catFields = dat.select_dtypes(['object','category'])

	if len(catFields.columns) > 0:
		for col in catFields.columns:
			means = dat.groupby(conditions)[col].mode()[0]
			dat = dat.set_index(conditions)
			dat[col] = dat[col].fillna(means)
			dat = dat.reset_index()

	if len(numFields.columns) > 0:
		for col in numFields.columns:
			means = dat.groupby(conditions)[col].mean()
			dat = dat.set_index(conditions)
			dat[col] = dat[col].fillna(means)
			dat = dat.reset_index()

	return dat

###############################################################
'''
Functions dealing with the actual pipeLine
'''

'''
Remove a key from a dictionary
'''
def removeKey(d, key):
    r = dict(d)
    del r[key]
    return r

'''
Get X and y from a dataframe
'''
def getXY(df, yName):
	y = df[yName]
	X = df.drop(yName, 1)
	return (y,X)

'''
Calls the scikit learn classifier function with
certain parameters.
'''
def wrapper(func, args):
	m = func(**args)
	return m

'''
Create dictionary from a sorted list of 
keys and items.
'''
#def createDict(keys, items):
#	return dict(zip(keys, items))
	'''
	d = {}
	for k in range(0, len(keys)):
		d[keys[k]] = items[k]
	return d
	''' 

'''
Make all the requisite mini dictionaries from
the main dictionary.
'''
def makeDicts(d):
	result = []
	dN = removeKey(d, 'model')
	thingy  = dN.keys()
	l = [dN[x] for x in thingy]
	combos = list(itertools.product(*l))
	lengthy = len(combos)
	result = [0] * lengthy

	for i in range(0, lengthy):
		result[i] = dict(zip(thingy, combos[i]))#createDict(thingy, combos[i])

	result.append({})

	return result

'''
Determine the best model from a dictionary of specific parameters
'''
def makeModels(X,y, d):
	result = makeDicts(d)

	z = 0
	total = len(result)
	for item in result:
			wrap = wrapper(d['model'], item)
			temp = wrap.fit(X,y)
			result[z] = temp
			z +=1
			print str(z) + '/' + str(total)
	return result

'''
Get a list of accuracies from a model list
'''
def getAccuracies(X, y, modelList):
	result = [0]*len(modelList)

	for i in range(0, len(modelList)):
		result[i] = modelList[i].score(X,y)
	return result

'''
Sort the list of models from best to worst
according to a second list of accuracies
'''
def bestModels(modelList, accList, rev = True):
	result = [x for (y,x) in sorted(zip(accList, modelList))]
	if rev:
		result.reverse()
	return result

'''
Loop over a list of models and their parameters.
From that list, calculate the accuracy of each model.
Then based on that accuracy, get a list of the best models
with the best model in the 0th position.
'''
def pipeLine(name, lModels, yName, criterion = getAccuracies, rev = True, fillMethod = fillNaMean):
	data = readcsv(name)
	df = fillMethod(data)
	y,X = getXY(df, yName)
	allModels = []
	indx = 1

	for l in lModels:
		print "\nIter: " + str(indx) + "\n"
		models = makeModels(X,y, l)
		allModels += models
		indx += 1
	print "\nFinished making models. Moving on to calculating given criterion\n"

	criteria = criterion(X,y, allModels)
	print "\nPutting models in desired order based on given criterion\n"
	result = bestModels(allModels, criteria, rev)

	return (result, criteria)

'''
Calculate the accuracy of a model
'''
def accuracy(X,y, model):
	return model.score(X,y)

'''
Generate predictions from model given x
'''
def getPredicts(model, x):
	p = model.predict(x)
	return p

'''
num1 = 100
num2 = 200
numCores = 4
models = {'model': LogisticRegression, '0': {'solver': 'newton-cg'},
		  '2': {'solver': 'lbfgs'},
		  '4': {'solver' :'sag'}, '6': {'solver': 'liblinear'}}

data = readcsv('training.csv')
print descrTable(data)
data = fillNaMean(data)
print corrTable(data)

bModel, accs = pipeLine('training.csv', modelList, 'SeriousDlqin2yrs')



dTest = readcsv('test.csv')
dTest = fillNaMean(dTest)

yTest, xTest = getXY(dTest, 'SeriousDlqin2yrs')

preds = getPredicts(bModel, xTest)
pDF = pd.DataFrame(preds)
pDF.to_csv("predictions.csv")
'''