# Michael Fosco, mfosco
# 4/12/16

import numpy as np 
import pandas as pd 
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import os, timeit, sys
import itertools
import seaborn as sns
from sklearn import linear_model, neighbors, ensemble, svm



def camel_to_snake(column_name):
    '''
    converts from camel case to snake case
    Taken from:  http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    '''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def read_data(filename):
    """
    Takes the name of a file to be read, returns a DataFrame object.
    filename should be a string. 
    """
    assert(type(filename)==str and filename.endswith('.csv'))
    data = pd.read_csv(filename,index_col=0)
    data.columns = [camel_to_snake(col) for col in data.columns]
    return data

'''
Read data from a csv
'''
def readcsv(filename):
	assert(type(filename) == str and filename.endswith('.csv'))
	data = pd.read_csv(filename, index_col = 0)
	data.columns = [camel_to_snake(col) for col in data.columns]
	return data

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
		plt.clf()

'''
Make pie plots.
'''
def piePlots(df, items, saveExt = ''):
	for it in items:
		b = df[it].value_counts().plot(kind = 'pie', title = it)
		s = saveExt + it + 'Bar.pdf'
		b.get_figure().savefig(s)
		plt.clf()

'''
Discretize a continous variable. 
num: 		The number of buckets to split the cts variable into
'''
def discretize(df, col, num=10):
	dDF = df
	dDF[col] = pd.cut(dDF[col], num)
	return dDF

'''
Convert a categorical variable into binary variables
'''
def categToBin(df, col):
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


'''
Generate a density plot for certain items
'''
def densityPlots(df, items, saveExt = ''):
	for it in items:
		b = df[it].value_counts().plot(kind = 'kde', title = it + ' Density Plot')
		s = saveExt + it + 'DensityPlot.pdf'
		b.get_figure().savefig(s)	

'''
Fill in the mean for select items
'''
def fillNaMean(df, items = 'all'):
	tdf = df

	if items == 'all':
		for col in tdf.columns:
			tdf[col] = tdf[col].fillna(tdf[col].mean())
		return tdf

	for i in items:
		tdf[i] = tdf[i].fillna(tdf[i].mean())
	return tdf

'''
Fill in the conditional mean for select items
'''
def fillConditMean(df, conditions, items = 'all'):
	if type(conditions) != list:
		raise TypeError('Error in fillConditMean. Variable CONDITIONS was not of type list')

	dat = df
	if items == 'all':
		for col in dat.columns:
			means = dat.groupby(conditions)[col].mean()
			dat = dat.set_index(conditions)
			dat[col] = dat[col].fillna(means)
			dat = dat.reset_index()
		return dat

	for col in items:
		means = dat.groupby(conditions)[col].mean()
		dat = dat.set_index(conditions)
		dat[col] = dat[col].fillna(means)
		dat = dat.reset_index()
	return dat

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
Determine the best model from a dictionary of specific parameters
'''
def bestModel(X,y, d):
	dN = removeKey(d, 'model')
	bestModel = None
	bAccuracy = 0
	for k in dN.keys():
			wrap = wrapper(d['model'], dN[k])
			temp = wrap.fit(X,y)
			tAcc = temp.score(X,y)

			if tAcc > bAccuracy:
				bAccuracy = tAcc
				bestModel = temp
				print str(bAccuracy)
	return bestModel, bAccuracy

modelsLR = {'model': LogisticRegression, 'solver': ['liblinear'], 'C' : [.1, .5, 1, 5, 10, 25],
		  'class_weight': ['balanced', 'auto', None],
		  'tol' : [1e-5, 1e-4, 1e-3, 1e-1, 1]}

def createDict(keys, items):
	d = {}

	for k in range(0, len(keys)):
		d[keys[k]] = items[k]
	return d 


def makeDicts(d):
	result = []
	dN = removeKey(d, 'model')

	thingy  = dN.keys()
	l = [dN[x] for x in thingy]
	combos = list(itertools.product(*l))
	lengthy = len(combos)
	result = [0] * lengthy

	for i in range(0, lengthy):
		result[i] = createDict(thingy, combos[i])

	return result

'''
Determine the best model from a dictionary of specific parameters
'''
def makeModels(X,y, d):
	result = makeDicts(d)

	z = 0
	for item in result:
			wrap = wrapper(d['model'], item)
			temp = wrap.fit(X,y)
			result[z] = temp
			z +=1
			print str(z)
	return result

def getAccuracies(X, y, modelList):
	result = [0]*len(modelList)

	for i in range(0, len(modelList)):
		result[i] = modelList[i].score(X,y)
	return result

'''
Hopefully generic enough model to test a bunch of 
classifiers and pick the best one (I feel this will need some
	more work)
'''
def pipeLine(name, lModels, yName):
	data = readcsv(name)
	df = fillNaMean(data)

	y,X = getXY(df, yName)

	bModel = None
	bAcc = 0
	for l in lModels:
		tMod, tAcc = bestModel(X,y, l)
		if tMod != None and tAcc > bAcc:
			bModel = tMod
			bAcc = tAcc
			print "Current best accuracy: " + str(bAcc)
	return bModel

'''
Fit a model given X and y
'''
def fitModel(X,y, model):
	return model.fit(X,y)

'''
Calculate the accuracy of a model
'''
def accuracy(X,y, model):
	return model.score(X,y)

'''
Fit a logistics regression model (semi useless after pipeLine)
'''
def fitting(y, X):
	model = LogisticRegression()
	mod = fitModel(X, y, model)

	return mod

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

bModel = pipeLine('training.csv', [models], 'SeriousDlqin2yrs')



dTest = readcsv('test.csv')
dTest = fillNaMean(dTest)

yTest, xTest = getXY(dTest, 'SeriousDlqin2yrs')

preds = getPredicts(bModel, xTest)
pDF = pd.DataFrame(preds)
pDF.to_csv("predictions.csv")
'''