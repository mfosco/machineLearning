#Michael Fosco
#4-3-16

import numpy as np 
import pandas as pd 
import requests
import statsmodels.api as sm

'''
Problem A

 

The first task is to load the file and generate summary statistics for each field
 as well as probability distributions or histograms. 
 The summary statistics should include mean, median, mode, standard deviation, 
 as well as the number of missing values for each field.
 DONE

You will notice that a lot of students are missing gender values . 
Your task is to infer the gender of the student based on their name. 
Please use the API at www.genderize.io to infer the gender of each student and 
generate a new data file.

You will also notice that some of the other attributes are missing. Your task is 
to fill in the missing values for Age, GPA, and Days_missed using the following 
approaches:
Fill in missing values with the mean of the values for that attribute
Fill in missing values with a class-conditional mean (where the class is 
	whether they graduated or not).
Is there a better, more appropriate method for filling in the missing values? 
If yes, describe and implement it. 
You should create 2 new files with the missing values filled, one for each approach
 A, B, and C and submit those along with your code. 

Please submit the Python code for each of these tasks as well as the new data files for this assignment.
'''

# First part
data = pd.read_csv("mock_student_data.csv")
print data.describe()
print "Note the median is the 50%"
missingVals = len(data.index) - data.count()
print missingVals
data['State'] = data['State'].fillna(data.State.interpolate())

for l in data.columns:
	if l != 'ID':
		modey = data[l].mode()[0]
		print l + ' mode: ' + str(modey)

z = data.hist(layout=(2,2))
z[0][1].get_figure().savefig('hists.pdf')


def getGender(name):
	r = requests.get('https://api.genderize.io/?name=' + name)
	s = r.text

	beg = s.index('gender')+9
	en = s[beg:].index('"') + beg
	gender = s[beg:en]

	if gender == 'male':
		return "Male"
	return "Female"

def addGenders(dat):
	df = dat
	temp = data['Gender']

	for i in df.index:
		#print str(i)
		if type(temp[i]) != str and np.isnan(temp[i]):
			df.set_value(i, 'Gender', getGender(df.get_value(i, 'First_name')))
	return df




#part ii
dfGender = addGenders(data)
dfGender.to_csv('genderAdded.csv')

#part iii
def wayA(dat):
	aMean = dat.Age.mean()
	gpaMean = dat.GPA.mean()
	dmMean = dat.Days_missed.mean()
	df = dat

	df['Age'] = df['Age'].fillna(aMean)
	df['GPA'] = df['GPA'].fillna(gpaMean)
	df['Days_missed'] = df['Days_missed'].fillna(dmMean)

	return df

dfA = wayA(pd.read_csv("genderAdded.csv"))

dfA.to_csv("wayA.csv")

def wayB(dat):
	df = dat
	means = df.groupby(['Graduated'])['Age'].mean()
	df = df.set_index(['Graduated'])
	df['Age'] = df['Age'].fillna(means)
	df = df.reset_index()

	means = df.groupby(['Graduated'])['GPA'].mean()
	df = df.set_index(['Graduated'])
	df['GPA'] = df['GPA'].fillna(means)
	df = df.reset_index()

	means = df.groupby(['Graduated'])['Days_missed'].mean()
	df = df.set_index(['Graduated'])
	df['Days_missed'] = df['Days_missed'].fillna(means)
	df = df.reset_index()

	return df

dfB = wayB(pd.read_csv("genderAdded.csv"))
dfB.to_csv("wayB.csv")

'''
Instead of just taking the means of the values (conditional or nonconditional)
it should be better to condition on even more. So I condition on
whether or not someone graduated, their gender, and their state.
'''
def wayC(dat):
	df = dat

	listy = ['Age', 'GPA', 'Days_missed']
	for l in listy:
		means = df.groupby(['Graduated', 'Gender', 'State'])[l].mean()
		df = df.set_index(['Graduated', 'Gender', 'State'])
		df[l] = df[l].fillna(means)
		df = df.reset_index()
	
	return df	

dfC = wayC(pd.read_csv("genderAdded.csv"))
dfC.to_csv("wayC.csv")













