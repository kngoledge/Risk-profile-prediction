import os, random, operator, sys
import collections
import math
import pandas as pd
import numpy as np
import csv
import re

############################################################
##  Related to data preparation
############################################################

def prepare_raw_WB_project_data():
	wb_small = pd.read_csv('WBsubset.csv')
	wb_small = wb_small[['sector1','sector2', 'sector3', 'sector4', 'sector5', 'sector', 'mjsector1','mjsector2', 'mjsector3', 'mjsector4', 'mjsector5', 'mjsector','Country','project_name']]
	return wb_small.fillna('')


def clean_sector_string(sector_string):
	sector_string = str(sector_string)
	if sector_string == 'nan':
		return []
	to_return = []
	sec = sector_string.split(';')
	for y in sec:
		to_add = re.sub(r'!\$!\d*!\$!\D\D', "", y)
		to_add = re.sub(r'\(.*\)', "", to_add)
	if to_add:
		if to_add[0] == " ":
			to_add = to_add[1:]
		to_return.append(to_add)
	return to_return


def prepare_clean_WB_project_data(df):
        """
                Returns a list of tuples, where the tuples are
                (project_name, [countries], [sectors]) for every datapoint
        """
	clean_data = []
	for index, x in df.iterrows():
		clean_sectors = clean_sector_string(x['sector1'])+clean_sector_string(x['sector2'])+clean_sector_string(x['sector3'])+clean_sector_string(x['sector4'])+clean_sector_string(x['sector5'])+clean_sector_string(x['sector'])+clean_sector_string(x['mjsector1'])+clean_sector_string(x['mjsector2'])+clean_sector_string(x['mjsector3'])+clean_sector_string(x['mjsector4'])+clean_sector_string(x['mjsector5'])+clean_sector_string(x['mjsector'])
		clean_countries = list(set(x['Country'].split(';')))
		clean_tuple = (x['project_name'], clean_countries, clean_sectors)
		clean_data.append(clean_tuple)
	return clean_data



def prepare_raw_complaint_data(): 
        """ 
                Slice relevant columns for country, sector, and issue
                and returns master complaint csv for project purposes.
        """
        df = pd.read_csv('complaints.csv')
        df = df[['Country', 'Sector/Industry (1)','Sector/Industry (2)',
                 'Issue Raised (1)','Issue Raised (2)', 'Issue Raised (3)', 
                 'Issue Raised (4)','Issue Raised (5)', 'Issue Raised (6)', 
                 'Issue Raised (7)', 'Issue Raised (8)', 'Issue Raised (9)', 
                 'Issue Raised (10)']]
        return df.fillna('')



def prepare_clean_complaint_data(df):
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

def get_project_names(): 
    ac = pd.read_csv('2016_17_complaints.csv')
    return list(set(filter(None, list(ac['Project Name'].fillna('')))))

def remove_duplicate_projects(proj_names, wb_data):
	"""
		proj_names is list of project names from COMPLAINTS
		wb_data is list of tuples where each element is tuple of (project name, [countries], [sectors])
		updates wb_data to remove instances of matching project names
	"""
	unmatched_data = []

	for tup in wb_data:
		match = False
		for name in proj_names:
			if tup[0] in name:
			#if tup[0] == name:
				match = True
				break
		if match == False: unmatched_data.append(tup)

	return unmatched_data

def combine_datasets(complaint_data, WB_data, numIssues):
	"""
		Returns a shuffled combo of complaint_data and unique WB_data
		Complaint Data: list of ([countries], [sectors], [issues]) tuples
		WB Data:        list of (proj name, [countries], [sectors]) tuples
	"""	
	#print('orig 0:', complaint_data[0], 'orig 20:', complaint_data[20], 'orig 600:', complaint_data[600])
	for a, b, c in WB_data:
		complaint_data.append( (b, c, ['NONE']) )
	random.shuffle(complaint_data)
	#print('new 0:', complaint_data[0], 'new 20:', complaint_data[20], 'new 600:', complaint_data[600])

	return complaint_data

############################################################
##  Related to featurization
############################################################

def build_dict(filename):
	""" 
		Returns a tuple consisting of
		(1) the list of regions
		(2) a dictionary mapping countries to regions
	"""
	your_list = []
	with open(filename) as inputfile:
		for row in csv.reader(inputfile):
			your_list.append(row[0])

	your_list = your_list[1:]

	dictionary = {}		# maps country to region
	categories = []		# list of regions
	category = ''

	for item in your_list:
		if item[0] == '*':
			category = item[1:]
			dictionary[category] = category
			categories.append(category)

		else:
			dictionary[item] = category

	return categories, dictionary

def featurize(inputList, featureVec):
	"""
	Converts a list of string inputs (sectors) into an extracted 
	feature vector, based on the related feature vector.
	Outputs a sparse feature vector.
	"""
	newVec = np.zeros(len(featureVec))

	for s in inputList:
		for i in range(len(featureVec)):
			if s == featureVec[i]: 
				newVec[i] = 1
				break

	return newVec.tolist()

def featurize_complex(inputList, featureVec, dictionary):
	"""
	Converts a list of string inputs (countries or sectors) into an
	extracted feature vector (based on regions or standard sectors).
	Outputs a sparse feature vector (of regions or standard sectors).
	"""
	#print "SUCCESSFUL COMPLEX FEATURIZER"
	#print("\nInput List: ", inputList)
	newVec = np.zeros(len(featureVec))

	for s in inputList:
		if s in dictionary:
			for i in range(len(featureVec)):
				if dictionary[s] == featureVec[i]: 
					newVec[i] = 1

	return newVec.tolist()

def featurize_issue(inputList, featureVec):
	"""
	Converts a list of string inputs (issues) into an extracted 
	feature vector, based on the related feature vector.
	If no issues, then the 'NONE' element is marked true.
	Outputs a sparse feature vector.
	"""
	newVec = np.zeros(len(featureVec))
	numIssues = 0

	for s in inputList:
		for i in range(len(featureVec) - 1):
			if s == featureVec[i]: 
				newVec[i] = 1
				numIssues += 1
				break

	if numIssues == 0: featureVec[-1] = 1
	return newVec.tolist()

# def featurize_issue(inputList, featureVec):
# 	"""
# 	Converts a list of string inputs (issues) into an extracted 
# 	feature vector, based on the related feature vector.
# 	If no issues, then the 'NONE' element is marked true.
# 	Outputs a sparse feature vector.
# 	"""

# 	issueDict = { 
# 		'Other retaliation (actual or feared)': 'Violence',
# 		'Livelihoods': 'Community',
# 		'Labor': 'Malpractice',
# 		'Consultation and disclosure': 'Malpractice',
# 		'Property damage': 'Community',
# 		'Indigenous peoples': 'Community',
# 		'Cultural heritage': 'Community',
# 		'Personnel issues': 'Malpractice',
# 		'Water': 'Environmental',
# 		'Other gender-related issues': 'Violence',
# 		'Biodiversity': 'Environmental',
# 		'Procurement': 'Malpractice',
# 		'Gender-based violence': 'Violence',
# 		'Other community health and safety issues': 'Community',
# 		'Pollution': 'Environmental',
# 		'Human rights': 'Violence',
# 		"Violence against the community (by gov't and/or company)": 'Violence',
# 		'Due diligence': 'Malpractice',
# 		'Displacement (physical and/or economic)': 'Community',
# 		'Other environmental': 'Envrionmental',
# 		'Corruption/fraud': 'Malpractice',
# 		'Other': 'Other',
# 		'NONE': 'NONE'
# 	}

# 	newVec = np.zeros(len(featureVec))
# 	numIssues = 0

# 	for s in inputList:
# 		if s in issueDict:
# 			for i in range(len(featureVec)-1):
# 				if issueDict[s] == featureVec[i]: 
# 					newVec[i] = 1
# 					numIssues += 1
# 					break

# 	if numIssues == 0: featureVec[-1] = 1
# 	newVec = np.zeros(len(featureVec))
# 	return newVec.tolist()


############################################################

def organize_data():
	"""
	Returns a list of inputs (x) and outputs (y) for the entire combined dataset.
	You must partition train vs. test & convert lists to arrays within neuralnet function. 
	"""

	regions, regionDict = build_dict('countrylist.csv')
	sectors, sectorDict = build_dict('sectorlist.csv')
	#sectors = ['Agribusiness', 'Infrastructure', 'Conservation and environmental protection', 'Energy', 'Healthcare', 'Manufacturing', 'Community capacity and development', 'Forestry', 'Chemicals', 'Other', 'Regulatory Development', 'Land reform', 'Education', 'Extractives (oil/gas/mining)']
	issues = ['Biodiversity', 'Consultation and disclosure', 'Corruption/fraud', 'Cultural heritage', 'Displacement (physical and/or economic)', 'Due diligence', 'Gender-based violence', 'Human rights', 'Indigenous peoples', 'Labor', 'Livelihoods', 'Other', 'Other community health and safety issues', 'Other environmental', 'Other gender-related issues', 'Other retaliation (actual or feared)', 'Personnel issues', 'Pollution', 'Procurement', 'Property damage', 'Unknown', "Violence against the community (by gov't and/or company)", 'Water', 'NONE']
	# add no issues to feature vector: will have to create a specialized featurize function


	numRegions = len(regions)
	numSectors = len(sectors)
	numIssues = len(issues)

	print('Num of Regions is ', numRegions)
	print('Num of Sectors is ', numSectors)	
	print('Num of Issues is ', numIssues)


	# issueGroups = ['Community', 'Environmental', 'Malpractice', 'Other', 'NONE']	# DELETE LATER IF FAIL
	# numGroups = len(issueGroups)
	# print('Num of Issue Groups is ', numGroups)

	complaint_df = prepare_raw_complaint_data()
	complaint_clean_df = prepare_clean_complaint_data(complaint_df)
	WB_df = prepare_raw_WB_project_data()
	WB_clean_df = prepare_clean_WB_project_data(WB_df)
	unique_WB_data = remove_duplicate_projects(get_project_names(), WB_clean_df)
	total_dataset = combine_datasets(complaint_clean_df, unique_WB_data, numIssues)

	print('Length of raw complaint: ', len(complaint_df) )
	print('Length of raw WB data: ', len(WB_df) )
	print('Length of unique WB data: ', len(unique_WB_data) )
	print('Length of total dataset: ', len(total_dataset) )

	## Convert data
	xlist = []
	ylist = []
	trainExamples = total_dataset
	for i in range(len(total_dataset)):
		x = featurize_complex(trainExamples[i][0], regions, regionDict)+featurize_complex(trainExamples[i][1], sectors, sectorDict)
		#x = featurize_complex(trainExamples[i][0], regions, regionDict)+featurize(trainExamples[i][1], sectors)
		y = featurize_issue(trainExamples[i][2], issues)
		#y = featurize_issue(trainExamples[i][2], issueGroups)	# DELETE LATER IF FAIL
		xlist.append(x)
		ylist.append(y)

	return xlist, ylist, numRegions, numSectors, numIssues

	#return xlist, ylist, numRegions, numSectors, numGroups



