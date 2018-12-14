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

MAX_MONEY = 1.5*10**9    # financial info, for reference
MIN_MONEY = 10**4
NUM_MONEY_BUCKETS = 10

def prepare_raw_WB_project_data():
	wb_small = pd.read_csv('WBsubset.csv')
	wb_small = wb_small[['sector1','sector2', 'sector3', 'sector4', 'sector5', 'sector', 'mjsector1','mjsector2', 'mjsector3', 'mjsector4', 'mjsector5', 'mjsector','Country','project_name', 'totalamt', 'grantamt']]
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
		(project_name, [countries], [sectors], finance) for every datapoint
	"""
	clean_data = []
	for index, x in df.iterrows():
		clean_sectors = clean_sector_string(x['sector1'])+clean_sector_string(x['sector2'])+clean_sector_string(x['sector3'])+clean_sector_string(x['sector4'])+clean_sector_string(x['sector5'])+clean_sector_string(x['sector'])+clean_sector_string(x['mjsector1'])+clean_sector_string(x['mjsector2'])+clean_sector_string(x['mjsector3'])+clean_sector_string(x['mjsector4'])+clean_sector_string(x['mjsector5'])+clean_sector_string(x['mjsector'])
		clean_countries = list(set(x['Country'].split(';')))
		clean_money = int(x['totalamt']) + int(x['grantamt'])
		clean_tuple = (x['project_name'], clean_countries, clean_sectors, clean_money)
		clean_data.append(clean_tuple)
	return clean_data


def prepare_2016_17_data(): 
	""" 
		Returns a list of tuples, where the tuples are 
		([countries], [sectors], [issues], finance) for every datapoint
	"""
	df = pd.read_csv('2016_17_complaints.csv')
	df = df.fillna('')
	df = pd.DataFrame({'Country': df['Country'], 'Finance': df['Money'], 'Issues': df['Issues'], 'Sector': df['Sector']})
	clean_data = [] 
	# clean data
	for index, x in df.iterrows():
		country = (x['Country'].split('/'))
		finance = (x['Finance'])
		issue = ([y.lstrip().rstrip() for y in str(x['Issues']).replace('Unknown', 'Other').replace('Extractives (oil, gas, mining)', 'Extractives (oil/gas/mining)').replace(', ', ',').replace(',', ';').split(';')])
		sector = (x['Sector'].split(';'))
		if country or finance or issue or sector:
			clean_tuple = (country, sector, issue,finance)
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

# def get_unique(column): 
# 	""" 
# 		Given a column from the master complaints df,
# 		return a list of its unique values
# 	"""
# 	u_column = []
# 	for x in column: 
# 		if x == x:
# 			for y in x.replace('Unknown', 'Other').replace('Extractives (oil, gas, mining)', 'Extractives (oil/gas/mining)').replace(', ', ',').split(','): 
# 				u_column.append(y)
# 	return list(set(u_column))

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
		Ignores project name.
	""" 
	#print('orig 0:', complaint_data[0], 'orig 20:', complaint_data[20], 'orig 600:', complaint_data[600])
	for a, b, c, d in WB_data:
		complaint_data.append( (b, c, ['NONE'], d) )
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

	dictionary = {}     # maps country to region
	categories = []     # list of regions
	category = ''

	for item in your_list:
		if item[0] == '*':
			category = item[1:]
			dictionary[category] = category
			categories.append(category)

		else:
			dictionary[item] = category

	return categories, dictionary


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

def featurize_finance(investment, average_money):
	"""
	Converts a single monetary investment into an extracted feature vector,
	where the amount is bucketed based on magnitude. 
	"""

	#newVec = [0, 0, 0, 0, 0, 0, 0, 0]
	newVec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

	if investment == 0: money = average_money
	else: money = investment

	if   money < 1*10**5: newVec[0]=1
	elif money < 5*10**5: newVec[1]=1
	elif money < 1*10**6: newVec[2]=1
	elif money < 5*10**6: newVec[3]=1
	elif money < 1*10**7: newVec[4]=1
	elif money < 5*10**7: newVec[5]=1
	elif money < 1*10**8: newVec[6]=1
	elif money < 5*10**8: newVec[7]=1
	elif money < 1*10**9: newVec[8]=1
	else: newVec[9]=1

	return newVec

# def featurize_issue(inputList, featureVec):
#     """
#     Converts a list of string inputs (issues) into an extracted 
#     feature vector, based on the related feature vector.
#     If no issues, then the 'NONE' element is marked true.
#     Outputs a sparse feature vector.
#     """
#     newVec = np.zeros(len(featureVec))
#     numIssues = 0

#     for s in inputList:
#         for i in range(len(featureVec) - 1):
#             if s == featureVec[i]: 
#                 newVec[i] = 1
#                 numIssues += 1
#                 break

#     if numIssues == 0: featureVec[-1] = 1
#     return newVec.tolist()

def featurize_issue(inputList, issueBuckets):
	"""
	Converts a list of string inputs (issues) into an extracted 
	feature vector, based on the related feature vector.
	If no issues, then the 'NONE' element is marked true.
	Outputs a sparse feature vector.
	"""
	issueMap = { 
		'Other retaliation (actual or feared)': 'Violence',
		'Livelihoods': 'Community',
		'Labor': 'Malpractice',
		'Consultation and disclosure': 'Community',
		'Property damage': 'Damages',
		'Indigenous peoples': 'Community',
		'Cultural heritage': 'Community',
		'Personnel issues': 'Malpractice',
		'Water': 'Environment',
		'Other gender-related issues': 'Violence',
		'Biodiversity': 'Environment',
		'Procurement': 'Malpractice',
		'Gender-based violence': 'Violence',
		'Other community health and safety issues': 'Community',
		'Pollution': 'Environment',
		'Human rights': 'Violence',
		"Violence against the community (by gov't and/or company)": 'Violence',
		'Due diligence': 'Malpractice',
		'Displacement (physical and/or economic)': 'Displacement',
		'Other environmental': 'Environment',
		'Corruption/fraud': 'Malpractice',
		'Other': 'Other',
		'NONE': 'NONE'
	}

	newVec = np.zeros(len(issueBuckets))
	numIssues = 0

	for s in inputList:
		if s in issueMap:
			for i in range(len(issueBuckets)):
				if issueMap[s] == issueBuckets[i]: 
					newVec[i] = 1
					numIssues += 1
					break

	if numIssues == 0: newVec[-1] = 1
	return newVec.tolist()

# def organize_issues(complaint_data): 
# 	unique_issues = set()
# 	for country, issue, sector, finance in complaint_data:
# 		for i in issue:
# 			unique_issues.add(i)

############################################################

def organize_data():
	"""
	Returns a list of inputs (x) and outputs (y) for the entire combined dataset.
	You must partition train vs. test & convert lists to arrays within neuralnet function. 
	"""

	regions, regionDict = build_dict('countrylist.csv')
	sectors, sectorDict = build_dict('sectorlist.csv')
	#issues = ['Biodiversity', 'Consultation and disclosure', 'Corruption/fraud', 'Cultural heritage', 'Displacement (physical and/or economic)', 'Due diligence', 'Gender-based violence', 'Human rights', 'Indigenous peoples', 'Labor', 'Livelihoods', 'Other', 'Other community health and safety issues', 'Other environmental', 'Other gender-related issues', 'Other retaliation (actual or feared)', 'Personnel issues', 'Pollution', 'Procurement', 'Property damage', 'Unknown', "Violence against the community (by gov't and/or company)", 'Water', 'NONE']
	issueBuckets = ['Community', 'Damages', 'Displacement', 'Environment', 'Malpractice', 'Other', 'Violence', 'NONE']
	
	numRegions = len(regions)
	numSectors = len(sectors)
	#numIssues = len(issues)
	numIssues = len(issueBuckets)		# 10 ISSUES BUCKETS
	numMoney = NUM_MONEY_BUCKETS


	print('#regions: ', numRegions)
	print regions
	print('#sectors: ', numSectors)
	print sectors
	print('#money: ', numMoney)
	print('#issues: ', numIssues)
	print issueBuckets

	complaint_data = prepare_2016_17_data()
	#organize_issues(complaint_data)
	print('Len Complaint Data: ', len(complaint_data))

	WB_df = prepare_raw_WB_project_data()
	WB_clean_df = prepare_clean_WB_project_data(WB_df)
	unique_WB_data = remove_duplicate_projects(get_project_names(), WB_clean_df)
	total_dataset = combine_datasets(complaint_data, unique_WB_data, numIssues)

	print('Len WB Unique Data: ', len(unique_WB_data))
	print('Len Total Data: ', len(total_dataset))

	sum_money = 0
	for i in range(len(unique_WB_data)):
		sum_money += unique_WB_data[i][3]
	average_money = sum_money / len(unique_WB_data)

	regionCounter = np.zeros(numRegions+numSectors+numMoney)
	bucketCounter = np.zeros(len(issueBuckets))

	## Convert data
	xlist = []
	ylist = []
	total_dataset = total_dataset
	for i in range(len(total_dataset)):
		x = featurize_complex(total_dataset[i][0], regions, regionDict)+featurize_complex(total_dataset[i][1], sectors, sectorDict)+featurize_finance(total_dataset[i][3], average_money)
		y = featurize_issue(total_dataset[i][2], issueBuckets)  # Sorted into 8 issues
		xlist.append(x)
		ylist.append(y)
		regionCounter += x
		bucketCounter += y
		
	print('Issue Distribution: ', bucketCounter, '\n Issue Buckets:', issueBuckets)
	return xlist, ylist, numRegions, numSectors, numIssues, numMoney



