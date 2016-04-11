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

'''
Read data from a csv
'''
def readcsv(filename):
	return pd.read_csv(filename)

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
def histPlots(df, items, num, fname, binns = 20, saveExt = ''):
	fig, axarr = plt.subplots(num, num)
	x = 0
	y = 0

	for it in items:
		makeHisty(axarr[x,y], df[it], it, binns)
		axarr[x,y].set_title(it)
		y += 1
		if y >= num:
			x += 1
			y = 0
		if x >= len(axarr):
			break
	fig.savefig(saveExt + fname + 'Hists.pdf')
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
This function will need some more work.
It attempts to call the scikit learn classifier function with
certain parameters.
'''
def wrapperLR(func, solver, fit_intercept = True):
	m = func( solver = solver, fit_intercept = fit_intercept)
	return m

'''
Determine the best model from a dictionary of specific parameters
'''
def bestModel(X,y, d):
	dN = removeKey(d, 'model')
	bestModel = None
	bAccuracy = 0
	for k in dN.keys():
			wrap = wrapperLR(d['model'], **dN[k])
			temp = wrap.fit(X,y)
			tAcc = temp.score(X,y)

			if tAcc > bAccuracy:
				bAccuracy = tAcc
				bestModel = temp
				print str(bAccuracy)
	return bestModel, bAccuracy

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