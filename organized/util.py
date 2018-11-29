import os, random, operator, sys
import collections
import math
import pandas as pd
import numpy as np
import csv

def regionInfo():
	""" 
		Returns a tuple consisting of
		(1) the list of regions
		(2) a dictionary mapping countries to regions
	"""
	your_list = []
	with open('countrylist.csv') as inputfile:
		for row in csv.reader(inputfile):
			your_list.append(row[0])

	your_list = your_list[1:]

	regionDict = {}		# maps country to region
	regions = []		# list of regions
	region = ''

	for country in your_list:
		if country[0] == '*':
			region = country[1:]
			regionDict[region] = region
			regions.append(region)

		else:
			regionDict[country] = region

	return regions, regionDict

def prepare_raw_data(filename): 
	""" 
		Slice relevant columns for country, sector, and issue
		and returns master complaint csv for project purposes.
	"""
	df = pd.read_csv(filename)
	df = df[['Country', 'Sector/Industry (1)','Sector/Industry (2)',
		 'Issue Raised (1)','Issue Raised (2)', 'Issue Raised (3)', 
		 'Issue Raised (4)','Issue Raised (5)', 'Issue Raised (6)', 
		 'Issue Raised (7)', 'Issue Raised (8)', 'Issue Raised (9)', 
		 'Issue Raised (10)']]
	return df.fillna('')

def prepare_clean_data(df):
	""" 
		Returns a list of tuples, where the tuples are 
		([countries], [sectors], [issues]) for every datapoint
	"""
	clean_data = [] 
	for index, x in df.iterrows():
		clean_sectors = filter(None,x['Sector/Industry (1)'].split('|')+x['Sector/Industry (2)'].split('|'))
		clean_issues = filter(None,x['Issue Raised (1)'].split('|')+x['Issue Raised (2)'].split('|')+x['Issue Raised (3)'].split('|')+x['Issue Raised (4)'].split('|')+x['Issue Raised (5)'].split('|')+x['Issue Raised (6)'].split('|')+x['Issue Raised (7)'].split('|')+x['Issue Raised (8)'].split('|')+x['Issue Raised (9)'].split('|')+x['Issue Raised (10)'].split('|'))
		clean_tuple = (x['Country'].split('|'), clean_sectors, clean_issues)
		clean_data.append(clean_tuple)
	return clean_data 

def get_unique(column): 
	""" 
		Given a column from the master complaints df,
		return a list of its unique values
	"""
	u_column = []
	for x in column: 
		if x == x:
			for y in x.replace('Unknown', 'Other').replace('Extractives (oil, gas, mining)', 'Extractives (oil/gas/mining)').replace(', ', ',').split(','): 
				u_column.append(y)
	return list(set(u_column))

def featurize(inputList, featureVec):
	"""
	Converts a list of string inputs (sectors or issues) into an
	extracted feature vector, based on the related feature vector.
	Outputs a sparse feature vector.
	"""
	newVec = np.zeros(len(featureVec))

	for s in inputList:
		for i in range(len(featureVec)):
			if s == featureVec[i]: 
				newVec[i] += 1
				break

	return newVec.tolist()

def featurize_country(inputList, featureVec, dictionary):
	"""
	Converts a list of string inputs (countries) into an
	extracted feature vector (based on regions).
	Outputs a sparse feature vector of regions.
	"""
	newVec = np.zeros(len(featureVec))

	for s in inputList:
		if s in dictionary:
			for i in range(len(featureVec)):
				if dictionary[s] == featureVec[i]: 
					newVec[i] += 1
					break

	return newVec.tolist()


############################################################

def organize_data(filename, numTrainers):
	"""
	Converts a list of string inputs (countries) into an
	extracted feature vector (based on regions).
	Outputs a sparse feature vector of regions.
	"""

	regions = regionInfo()[0]
	regionDict = regionInfo()[1]
	sectors = ['Agribusiness', 'Infrastructure', 'Conservation and environmental protection', 'Energy', 'Healthcare', 'Manufacturing', 'Community capacity and development', 'Forestry', 'Chemicals', 'Other', 'Regulatory Development', 'Land reform', 'Education', 'Extractives (oil/gas/mining)']
	issues = ['Other retaliation (actual or feared)', 'Livelihoods', 'Labor', 'Consultation and disclosure', 'Property damage', 'Other', 'Indigenous peoples', 'Cultural heritage', 'Personnel issues', 'Water', 'Other gender-related issues', 'Biodiversity', 'Procurement', 'Gender-based violence', 'Other community health and safety issues', 'Pollution', 'Human rights', "Violence against the community (by gov't and/or company)", 'Due diligence', 'Displacement (physical and/or economic)', 'Other environmental', 'Corruption/fraud']

	numRegions = len(regions)
	numSectors = len(sectors)
	numIssues = len(issues)

	df = prepare_raw_data(filename)
	clean_df = prepare_clean_data(df)

	## Training Data
	xtrain = []
	ytrain = []
	trainExamples = clean_df[:numTrainers]
	for i in range(numTrainers):
		x = featurize_country(trainExamples[i][0], regions, regionDict)+featurize(trainExamples[i][1], sectors)
		y = featurize(trainExamples[i][2], issues)
		xtrain.append(x)
		ytrain.append(y)
	xtrain = np.asarray(xtrain)
	ytrain = np.asarray(ytrain)

	## Testing Data
	xtest = []
	ytest = []
	testExamples = clean_df[numTrainers:]
	numTesters = len(testExamples)      #change to len(testExamples)
	for i in range(numTesters):
		x = featurize_country(testExamples[i][0], regions, regionDict)+featurize(trainExamples[i][1], sectors)
		y = featurize(trainExamples[i][2], issues)
		xtest.append(x)
		ytest.append(y)
	xtest = np.asarray(xtest)
	ytest = np.asarray(ytest)

	return xtrain, ytrain, xtest, ytest, numRegions, numSectors, numIssues



