# Michael Fosco, mfosco
# 5/6/16

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
import os, timeit, sys, itertools, re, time, requests, random, functools, logging, csv, datetime
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
Note: model.predict( x) predicts from model using x
'''
##################################################################################
'''
List of models and parameters
'''
criteriaHeader = ['AUC', 'Accuracy', 'Function called', 'Precision at .05',
 				  'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5',
 				  'Precision at .75','Precision at .85','Recall at .05','Recall at .10',
 				  'Recall at .20','Recall at .25','Recall at .5','Recall at .75',
 				  'Recall at .85','f1 at 0.05','f1 at 0.1','f1 at 0.2','f1 at 0.25',
 				  'f1 at 0.5','f1 at 0.75','f1 at 0.85','test_time (sec)','train_time (sec)']

modelNames = ['LogisticRegression', 'KNeighborsClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier',
			  'AdaBoostClassifier', 'SVC', 'GradientBoostingClassifier', 'GaussianNB', 'DecisionTreeClassifier',
			  'SGDClassifier']
n_estimMatrix = [5, 10, 25, 50, 100, 200]#, 1000, 10000]
depth = [10, 20, 50]# 100]#1, 5, 100
cpus = mp.cpu_count()
cores = cpus-1
modelLR = {'model': LogisticRegression, 'solver': ['liblinear'], 'C' : [.01, .1, .5, 1],#, 5, 10, 25],
		  'class_weight': ['balanced', None], 'n_jobs' : [cores],
		  'tol' : [1e-5, 1e-3, 1], 'penalty': ['l1', 'l2']} #tol also had 1e-7, 1e-4, 1e-1
#took out linear svc because it did not have predict_proba function
#modelLSVC = {'model': svm.LinearSVC, 'tol' : [1e-7, 1e-5, 1e-4, 1e-3, 1e-1, 1], 'class_weight': ['balanced', None],
#			 'max_iter': [1000, 2000], 'C' :[.01, .1, .5, 1, 5, 10, 25]}
modelKNN = {'model': neighbors.KNeighborsClassifier, 'weights': ['uniform', 'distance'], 'n_neighbors' : [100, 500, 1000],#2,5, 10, 50, 10000],
			'leaf_size': [60, 120], 'n_jobs': [cpus/4]} #leaf size also had 15, 30
modelRF  = {'model': RandomForestClassifier, 'n_estimators': [25, 50, 100], 'criterion': ['gini', 'entropy'],
			'max_features': ['sqrt', 'log2'], 'max_depth': depth, 'min_samples_split': [20, 50], #min sample split also had 2, 5, 10
			'bootstrap': [True], 'n_jobs':[cores]} #bootstrap also had False
#have to redo just the one below
modelET  = {'model': ExtraTreesClassifier, 'n_estimators': [25, 50, 100], 'criterion': ['gini', 'entropy'],
			'max_features': ['sqrt', 'log2'], 'max_depth': depth,
			'bootstrap': [True, False], 'n_jobs':[cores]}
#base classifier for adaboost is automatically a decision tree
modelAB  = {'model': AdaBoostClassifier, 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [5, 10, 25, 50, 100]}#, 200]}
modelSVM = {'model': svm.SVC, 'C':[0.1,1], 'max_iter': [1000, 2000], 'probability': [True], 
			'kernel': ['rbf', 'poly', 'sigmoid', 'linear']} #C was: [0.00001,0.0001,0.001,0.01,0.1,1,10]

# will have to change n_estimators when running this on the project data
#modelGB  = {'model': GradientBoostingClassifier, 'learning_rate': [0.01,0.05,], 'n_estimators': [1,10,50],#100], #200,1000,10000], these other numbers took way too long to calc, learning rate had .1 and .5
#			'max_depth': [5,10], 'subsample' : [0.1, .2]} #subsample included .5, 1, learning also had .001, 
#Naive Bayes below
modelNB  = {'model': GaussianNB}
modelDT  = {'model': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [10,20,50], #had 100, 1, 5
			'max_features': ['sqrt','log2'],'min_samples_split': [10, 20, 50]} #had a 2,5
modelSGD = {'model': SGDClassifier, 'loss': ['modified_huber', 'perceptron'], 'penalty': ['l1', 'l2', 'elasticnet'], 
			'n_jobs': [cores]}

modelList = [modelLR, modelKNN, modelRF, modelET, 
			 modelAB, modelSVM, modelNB, modelDT,
			 modelSGD] #had modelGB

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
		s = saveExt + it + 'Pie.pdf'
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
Make histogram plots, num is for layout of plot
'''
def histPlots(df, cols, numBins = 50):
	for col in cols:
		b = plt.hist(df[col], bins = numBins)
		plt.title(col)
		s = col + '_Hist.pdf'
		plt.savefig(s)
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
	try:
		m = func(**args)
		return m
	except:
		return None

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
		result[i] = dict(zip(thingy, combos[i]))

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
Return a result string as "mean (std)"
'''
def makeResultString(mean, std):
	return str(mean) + ' (' + str(std) + ')' 

'''
Turn prediction probabilities into 1s or 0s based
on a threshold, thresh.
'''
def getPredsAtThresh(thresh, predProbs):
	res = [None]*len(predProbs)
	indx = 0
	for z in predProbs:
		res[indx] = np.asarray([1 if j >= thresh else 0 for j in z])
		indx +=1

	return res

'''
Return a dictionary of a bunch of criteria. Namely, this returns a dictionary
with precision and recall at .05, .1, .2, .25, .5, .75, AUC, time to train, and
time to test.
'''
def getCriterions(yTests, predProbs, train_times, test_times, accuracies, called):
	levels = ['Precision at .05', 'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5', 'Precision at .75', 'Precision at .85']
	recalls = ['Recall at .05', 'Recall at .10', 'Recall at .20', 'Recall at .25', 'Recall at .5', 'Recall at .75', 'Recall at .85']
	amts= [.05, .1, .2, .25, .5, .75, .85]
	tots = len(amts)
	res = {}
	critsLen = len(yTests)
	critsRange = range(0, critsLen)
	res['Function called'] = called
	for x in xrange(0, tots):
		thresh = amts[x]
		preds = getPredsAtThresh(thresh, predProbs)
		prec = [metrics.precision_score(yTests[j], preds[j]) for j in critsRange]
		rec = [metrics.recall_score(yTests[j], preds[j]) for j in critsRange]
		precStd = np.std(prec)
		recStd = np.std(rec)

		f1S = [2*(prec[j]*rec[j])/(prec[j]+rec[j]) for j in critsRange]
		f1Std = np.std(f1S)

		precM = np.mean(prec)
		recM = np.mean(rec)
		f1M = np.mean(f1S)
		res[levels[x]] = makeResultString(precM, precStd)
		res[recalls[x]] = makeResultString(recM, recStd)
		res['f1 at ' + str(thresh)] = makeResultString(f1M, f1Std)

	auc = [metrics.roc_auc_score(yTests[j], predProbs[j]) for j in critsRange]
	aucStd = np.std(auc)
	aucM = np.mean(auc)
	trainM = np.mean(train_times)
	trainStd = np.std(train_times)
	testM = np.mean(test_times)
	testStd = np.std(test_times)
	accM = np.mean(accuracies)
	accStd = np.std(accuracies)

	res['AUC'] = makeResultString(aucM, aucStd)
	res['train_time (sec)'] = makeResultString(trainM, trainStd)
	res['test_time (sec)'] = makeResultString(testM, testStd)
	res['Accuracy'] = makeResultString(accM, accStd)

	return res

'''
Wrapper type function for own parallelization.
This will get prediction probabilities as well as the results
of a host of criteria described in getCriterions.
'''
def paralleled(item, X, y, k, modelType):
	s = str(item)
	logging.info('Started: ' + s)
	try:
		trainTimes = [None]*k
		predProbs = [None]*k
		testTimes = [None]*k
		yTests = [None]*k
		accs = [None]*k 
		kf = cross_validation.KFold(len(y), k)
		indx = 0
		wrapped = wrapper(modelType, item)
		for train, test in kf:
			XTrain, XTest = X._slice(train, 0), X._slice(test, 0)
			yTrain, yTest = y._slice(train, 0), y._slice(test, 0)
			yTests[indx] = yTest

			start = time()
			fitting = wrapped.fit(XTrain, yTrain)
			t_time = time() - start
			trainTimes[indx] = t_time
			start_test = time()
			predProb = fitting.predict_proba(XTest)[:,1]
			test_time = time() - start_test
			testTimes[indx] = test_time
			predProbs[indx] = predProb
			accs[indx] = fitting.score(XTest,yTest)
			indx +=1

		criteria = getCriterions(yTests, predProbs, trainTimes, testTimes, accs, str(wrapped))
	except:
		logging.info('Error with: ' + s)
		return None
	return criteria

'''
Same function as makeModels (below), but uses own parallelization.
This was written for the cases where sklearn did not have an 
n_jobs option and was not automatically parallelized. This will write 
the status of the parallelization to the file: status.log. 
'''
def makeModelsPara(X, y, k, d):
	global cores
	result = makeDicts(d)

	logging.info('\nStarted: ' + str(d['model']) + "\n")
	pool = mp.Pool(cores)
	res = pool.map(functools.partial(paralleled, X = X, y = y, k = k, modelType = d['model']), result)
	pool.close()
	pool.join()
	logging.info('\nEnded: ' + str(d['model']) + "\n")

	return res

'''
Fit a model and determine the results of a bunch of
criteria, namely precision at various levels and AUC.
'''
def makeModels(X, y, k, d):
	result = makeDicts(d)
	total = len(result)
	res = [None]*total

	z = 0
	kf = cross_validation.KFold(len(y), k)
	logging.info("\nStarting: " + str(d['model']) + '\n')
	for item in result:
			wrap = wrapper(d['model'], item)
			try:
				trainTimes = [None]*k
				predProbs = [None]*k
				testTimes = [None]*k
				yTests = [None]*k
				accs = [None]*k
				
				indx = 0
				for train, test in kf:
					XTrain, XTest = X._slice(train, 0), X._slice(test, 0)
					yTrain, yTest = y._slice(train, 0), y._slice(test, 0)
					yTests[indx] = yTest

					start = time()
					fitting = wrap.fit(XTrain, yTrain)
					t_time = time() - start
					trainTimes[indx] = t_time
					start_test = time()
					predProb = fitting.predict_proba(XTest)[:,1]
					test_time = time() - start_test
					testTimes[indx] = test_time
					predProbs[indx] = predProb
					accs[indx] = fitting.score(XTest,yTest)
					indx += 1

				criteria = getCriterions(yTests, predProbs, trainTimes, testTimes, accs, str(wrap))

				res[z] = criteria 
			except:
				print "Invalid params: " + str(item)
				continue
			z +=1
			s= str(z) + '/' + str(total)
			logging.info(s)
			print s
	logging.info("\nEnded: " + str(d['model']) + '\n')
	return res

'''
Retrieve criteria from results of pipeline.
No longer needed after adding k-fold cross val
def retrieveCriteria(results):
	fin = [x[1] for x in results if x[1] != None]
	return fin
'''

'''
Format the data to be in nice lists in the same 
order as the masterHeader.
'''
def formatData(masterHeader, d):
	length = len(d)
	format = [[]] * length
	lenMH = len(masterHeader)

	indxForm = 0
	indx = 0
	for x in d:
		tmp = [None] * lenMH
		for j in masterHeader:
			tmp[indx] = x[j]
			indx += 1
		indx = 0
		format[indxForm] = tmp
		indxForm += 1
	return format

'''
Write results of pipeline to file. Note, d is the 
variable that is returned by the pipeLine function call
Return:					0 for successful termination, -1 for error 
'''
def writeResultsToFile(fName, d):
	header = makeHeader(d)
	fin = formatData(header, d)
	fin.insert(0, header)
	try:
		with open(fName, "w") as fout:
			writer = csv.writer(fout)
			for f in fin:
				writer.writerow(f)
			fout.close()
	except:
		return -1
	return 0

'''
General pipeline for ML process. This reads data from a file and generates a 
training and testing set from it. It then fits a model and gets the models precision
at .05, .1, .2, .25, .5, .75, .85, and AUC. It returns a list of models fit as 
well as those model's results for each criterion.
'''
def pipeLine(name, lModels, yName, k, fillMethod = fillNaMean):
	data = readcsv(name)
	df = fillMethod(data)
	y,X = getXY(df, yName)
	res = []
	indx = 1
	logging.basicConfig(filename='status.log',level=logging.DEBUG)
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	for l in lModels:
		print "\nIter: " + str(indx) + "\n"
		#own parallelization if sklearn does not already do that
		if 'n_jobs' not in l and "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>" not in l.values():
			models = makeModelsPara(X, y, k, l)
			res += models
			#normalize data in case of KNN
		elif "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>" in l.values():
			Xtmp  = preprocessing.scale(X)
			models = makeModels(Xtmp, y, k, l)
			res += models 
		else:
			models = makeModels(X, y, k, l)
			res += models 

		indx +=1

	return [z for z in res if z != None]

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
Creates same header as what is written to file.
Variable d is what is outputted from pipeline.
'''
def makeHeader(d):
	header = None
	spot = 0
	limit = len(d)
	while(header == None and spot < limit):
		if d[spot] != None:
			header = [x for x in d[spot].keys()]

	if header == None:
		raise Exception('Unable to grab appropriate header. Please check pipeLine')

	header.sort()
	return header

'''
Puts the results from pipeline into lists.
The results are in the same order as they are written
to file.
'''
def getResultsInList(d):
	global criteriaHeader
	fin = formatData(criteriaHeader, d)
	return fin

'''
Determine if list l contains a string.
ex:		l = ['hello', 'good day']
		containsString('hello', l) -> True
		containsString('good', l)  -> True
		containsString('bad', l)   -> False
'''
def containsString(string, l):
	for x in l:
		if string in x:
			return True 
	return False

'''
Get all results from function call getResultsInList
by a given modelType. The order of result types will be the same
as what is written to file.
'''
def getResultsByModel(results, modelName):
	fin = []

	for item in results:
		if containsString(modelName, item):
			fin.append(item)

	return fin

'''
Read in file as a list.
'''
def readInAsList(fname):
	try:
		with open(fname, 'rb') as fIn:
			reader = csv.reader(f)
			return list(reader)
	except:
		return -1

'''
Write a list to file.
header:				Header of file to be written (is a list).
data:				Data of file to be written (is a list)
fName:				Name of file to be written.
'''
def writeListToFile(header, data, fName):
	assert(type(data) == list and type(header) == list)
	try:
		fin = data
		fin.insert(0, header)
		with open(fName, 'w') as fout:
			writer = csv.writer(fout)
			for f in fin:
				writer.writerow(f)
			fout.close()
		del fin 
	except:
		raise Exception('Error (writeListToFile) in writing list to file')

'''
Write n entries per metric to separate file for all models.
data:					Data of metrics as a list.
n:						Number of entries to get per model for each metric.
'''
def writeNMetricsFilePerModel(data, n):
	global criteriaHeader
	global modelNames
	header = criteriaHeader
	now = datetime.datetime.now() 
	datey = str(now.month) + "." + str(now.day) + "." + str(now.year)
	endPart = '_' + datey + '.csv'

	modelResults = [getResultsByModel(data, x) for x in modelNames]
	headLen = len(header)

	for x in xrange(headLen):
		metric = header[x]
		tmp = []
		if metric != 'Function called':
			if '(sec)' in metric:
				tmp = getBestNResultsAllModelsForSpecificMetric(data, modelResults, n, x, True)
				writeListToFile(header, tmp, metric + endPart)
			else:
				tmp = getBestNResultsAllModelsForSpecificMetric(data, modelResults, n, x, False)
				writeListToFile(header, tmp, metric + endPart)

'''
Gets best n results across all model types for a specific metric.
'''
def getBestNResultsAllModelsForSpecificMetric(data, modelResults, n, indx, lowest = False):
	res = []
	for z in modelResults:
		res += getNResultsForMetric(z, indx, n, lowest)


	final = mergeSort(res, indx)

	if not lowest:
		final.reverse()
		return final 
	return final

'''
Return top n results for metric at index, indx.
Results is not assumed to be sorted.
'''
def getNResultsForMetric(results, indx, n, lowest = False):
	sorted_res = mergeSort(results, indx)

	resLen = len(sorted_res)

	items = min(n, resLen)

	if resLen != 0:
		if not lowest:
			return sorted_res[resLen-items:]
		else:
			return sorted_res[0:items]

	return []

'''
Gets the result value at index, indx, in the 
list generated by getCriterions.
'''
def getResultValue(l, indx):
	try:
		s = l[indx]
		parensIndx = s.index('(')
		return float(s[0:parensIndx])
	except:
		raise Exception('Invalid format of passed list for mergeSort.\nL was: ' + str(l) + '. Index was: ' + str(indx))

'''
Performs mergesort on a list of results from getCriterions. Works in O(nlogn) time.
A: 		The list of results from getCriterions.
indx:   The index of desired metric to sort by.
'''
def mergeSort(A, indx):
    if len(A)>1:
        mid = len(A)//2
        lefthalf = A[:mid]
        righthalf = A[mid:]

        lefthalf = mergeSort(lefthalf, indx)
        righthalf = mergeSort(righthalf, indx)

        i=0
        j=0
        k=0
        leftLen = len(lefthalf)
        rightLen = len(righthalf)
        while i < leftLen and j < rightLen:
            if getResultValue(lefthalf[i], indx) < getResultValue(righthalf[j], indx):
                A[k]=lefthalf[i]
                i=i+1
            else:
                A[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < leftLen:
            A[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < rightLen:
            A[k]=righthalf[j]
            j=j+1
            k=k+1
    return(A)

'''
name = 'training.csv'
k = 5
yName = 'SeriousDlqin2yrs'
thingsToWrite = pipeLine(name, modelList, yName, k, fillNaMedian)
writeResultsToFile('resultsTable.csv', thingsToWrite)
'''