# Michael Fosco, mfosco
# 4/12/16

import numpy as np 
import pandas as pd 
from scipy import optimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
import multiprocessing as mp
from multiprocessing import Process, Queue
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import os, timeit, sys, itertools, re, time, requests, random, functools, logging
import seaborn as sns
from sklearn import linear_model, neighbors, ensemble, svm, preprocessing
from numba.decorators import jit, autojit
from numba import double #(nopython = True, cache = True, nogil = True [to run concurrently])
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from time import time 

'''
Notes from hw2: get rid of special cases for plots
just make one plot per param then decide how to arrange them in the report

need to impute test set using train set statistics
'''

'''
Note: model.predict( x) predicts from model using x
'''
##################################################################################
'''
List of models and parameters
'''

cores = mp.cpu_count()-1
modelLR = {'model': LogisticRegression, 'solver': ['liblinear'], 'C' : [.01, .1, .5, 1, 5, 10, 25],
		  'class_weight': ['balanced', None], 'n_jobs' : [cores],
		  'tol' : [1e-7, 1e-5, 1e-4, 1e-3, 1e-1, 1], 'penalty': ['l1', 'l2']}
modelLSVC = {'model': svm.LinearSVC, 'tol' : [1e-7, 1e-5, 1e-4, 1e-3, 1e-1, 1], 'class_weight': ['balanced', None],
			 'max_iter': [1000, 2000], 'C' :[.01, .1, .5, 1, 5, 10, 25]}
modelKNN = {'model': neighbors.KNeighborsClassifier, 'weights': ['uniform', 'distance'], 'n_neighbors' : [2, 5, 10, 50, 100, 500, 1000, 10000],
			'leaf_size': [15, 30, 60, 120], 'n_jobs': [cores]}
modelRF  = {'model': RandomForestClassifier, 'n_estimators': [5, 10, 25, 50, 100, 200, 1000, 10000], 'criterion': ['gini', 'entropy'],
			'max_features': ['sqrt', 'log2'], 'max_depth': [1, 5, 10, 20, 50, 100], 'min_samples_split': [2, 5, 10, 20, 50],
			'bootstrap': [True, False], 'n_jobs':[cores]}
modelET  = {'model': ExtraTreesClassifier, 'n_estimatores': [5, 10, 25, 50, 100, 200, 1000, 10000], 'criterion': ['gini', 'entropy'],
			'max_features': ['sqrt', 'log2'], 'max_depth': [1, 5, 10, 20, 50, 100],
			'bootstrap': [True, False], 'n_jobs':[cores]}
#base classifier for adaboost is automatically a decision tree
modelAB  = {'model': AdaBoostClassifier, 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1, 10, 100, 200, 1000, 10000]}
modelSVM = {'model': svm.SVC, 'C':[0.00001,0.0001,0.001,0.01,0.1,1,10], 'probability': [True], 'kernel': ['rbf', 'poly', 'sigmoid']}
modelGB  = {'model': GradientBoostingClassifier, 'learning_rate': [0.001,0.01,0.05,0.1,0.5], 'n_estimators': [1,10,100], #200,1000,10000], these other numbers took way too long to calc
 			'max_depth': [1,3,5,10,20,50,100], 'subsample' : [0.1, .2, 0.5, 1.0]}
#Naive Bayes below
modelNB  = {'model': GaussianNB}
modelDT  = {'model': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 
			'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10, 20, 50]}
modelSGD = {'model': SGDClassifier, 'loss': ['modified_huber', 'perceptron'], 'penalty': ['l1', 'l2', 'elasticnet'], 
			'n_jobs': [cores]}

modelList = [modelLR, modelLSVC, modelKNN, modelRF, modelET, 
			 modelAB, modelSVM, modelGB, modelNB, modelDT,
			 modelSGD]

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
def makeHisty(ax, col, it, binny = 20):
	n, bins, patches = ax.hist(col, binns=binny, histtype='bar', range=(min(col), max(col)))

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

def fillNaMedian(data):
    """
    fills in missing values using unconditional mode/median as appropriate
    """
    numeric_fields = data.select_dtypes([np.number])
    categorical_fields = data.select_dtypes(['object','category'])
    
    if len(categorical_fields.columns) > 0:
        for col in categorical_fields.columns:
            ind = pd.isnull(data[col])
            fill_val = data[col].mode()[0]
            data.ix[ind,col] = fill_val
        
    if len(numeric_fields.columns) > 0:    
        for col in numeric_fields.columns:
            ind = pd.isnull(data[col])
            fill_val = data[col].median()
            data.ix[ind,col] = fill_val

    return data
            


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
Remove a key from a dictionary. Used in makeDicts.
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
Wrapper for function, func, with arguments,
arg, coming in a dictionary.
'''
def wrapper(func, args):
	m = func(**args)
	return m


'''
Make all the requisite mini dictionaries from
the main dictionary for pipeline process.
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

	#result.append({})

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
Return a dictionary of a bunch of criteria. Namely, this returns a dictionary
with precision and recall at .05, .1, .2, .25, .5, .75, AUC, time to train, and
time to test.
'''
def getCriterions(yTest, yPredProbs, train_time, test_time):
	levels = ['Precision at .05', 'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5', 'Precision at .75']
	recalls = ['Recall at .05', 'Recall at .10', 'Recall at .20', 'Recall at .25', 'Recall at .5', 'Recall at .75']
	amts= [.05, .1, .2, .25, .5, .75]
	tots = len(amts)
	res = {}
	for x in xrange(0, tots):
		thresh = amts[x]
		prec = precision_at_k(yTest, yPredProbs, thresh)
		rec = metrics.recall_score(yTest, predictionsAtThresh(yTest, yPredProbs, thresh))
		res[levels[x]] = prec
		res[recalls[x]] = rec
		#I know there is an f1 sklearn function, but I believe this way is marginally faster
		res['f1 at ' + str(thresh)] = 2*((prec*rec)/(prec+rec))

	res['AUC'] = metrics.roc_auc_score(yTest, yPredProbs)
	res['train_time (sec)'] = train_time
	res['test_time (sec)'] = test_time
	return res

'''
Wrapper type function for own parallelization.
This will get prediction probabilities as well as the results
of a host of criteria described in getCriterions.
'''
def paralleled(item, XTrain, XTest, yTrain, yTest, modelType):
	logging.info(str(item))
	try:
		start = time()
		wrapped = wrapper(modelType, item)
		preds = wrapped.fit(XTrain, yTrain)
		t_time = time() - start
		start_test = time()
		predProb = preds.predict_proba(XTest)[:,1]
		test_time = time() - start_test
		criteria = getCriterions(yTest, predProb, t_time, test_time)
	except:
		logging.info('Error with: ' + str(item))
		return (None, None)
		
	return (wrapped, criteria)

'''
Same function as makeModels (below), but uses own parallelization.
This was written for the cases where sklearn did not have an 
n_jobs option and was not automatically parallelized. This will write 
the status of the parallelization to the file: status.log. 
'''
def makeModelsPara(XTrain, XTest,yTrain, yTest, d):
	global cores
	result = makeDicts(d)

	logging.basicConfig(filename='status.log',level=logging.DEBUG)
	logging.info('Started: ' + str(d['model']) + "\n")
	pool = mp.Pool(cores)
	res = pool.map(functools.partial(paralleled, XTrain = XTrain, XTest = XTest, yTrain = yTrain, yTest = yTest, modelType = d['model']), result)
	pool.close()
	pool.join()
	logging.info('Ended: ' + str(d['model']) + "\n")

	return res

'''
Fit a model and determine the results of a bunch of
criteria, namely precision at various levels and AUC.
'''
def makeModels(XTrain, XTest,yTrain, yTest, d):
	result = makeDicts(d)
	total = len(result)
	criterions = [None]*total

	z = 0
	for item in result:
			wrap = wrapper(d['model'], item)
			try:
				start = time()
				preds = wrap.fit(XTrain, yTrain)
				t_time = time() - start
				start_test = time()
				yPredProbs = preds.predict_proba(XTest)[:,1]
				test_time = time() - start_test
				criterions[z] = getCriterions(yTest, yPredProbs, t_time, test_time)
				result[z] = (wrap, yPredProbs)
			except:
				print "Invalid params: " + str(item)
				result[z] = None
				continue
			z +=1
			print str(z) + '/' + str(total)
	return (result, criterions)

'''
General pipeline for ML process. This reads data from a file and generates a 
training and testing set from it. It then fits a model and gets the models precision
at .05, .1, .2, .25, .5, .75, and AUC. It returns a list of models fit as well as 
those model's results for each criterion.
'''
def pipeLine(name, lModels, yName, fillMethod = fillNaMean):
	data = readcsv(name)
	df = fillMethod(data)
	y,X = getXY(df, yName)
	res = []
	indx = 1
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	for l in lModels:
		print "\nIter: " + str(indx) + "\n"
		#own parallelization if sklearn does not already do that
		if 'n_jobs' not in l and "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>" not in l.values():
			models = makeModelsPara(X_train, X_test, y_train, y_test, l)
			res += models
			#normalize data in case of KNN
		elif "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>" in l.values():
			nXTrain = preprocessing.scale(X_train)
			nXTest = preprocessing.scale(X_test)
			models = makeModels(nXTrain, nXTest, y_train, y_test, l)
			res += models 
		else:
			models = makeModels(X_train, X_test, y_train, y_test, l)
			res += models 

		indx +=1

	return res

'''
Plot precision recall curve given model, true y, and
predicted y probabilities.
'''
def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()

'''
Get predictions at threshold
'''
def predictionsAtThresh(y_true, y_scores, k):
	threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
	y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
	return y_pred


'''
Precision at certain cutoff value 
'''
def precision_at_k(y_true, y_scores, k):
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred)


'''
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