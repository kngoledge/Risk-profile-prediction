#!/usr/bin/python
import os, random, operator, sys
import collections
import math
import pandas as pd
import numpy as np

def prepare_raw_data(): 
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

############################################################

def featurize(inputList, featureVec):
    """
    Converts string input (sectors or countries or issues) into an
    extracted feature vector, based on the related feature vector.
    Outputs a sparse feature vector that is the concatenation of
    the sector feature vec followed by the country feature vec. 
    """
    newVec = np.zeros(len(featureVec))

    for i in range(len(featureVec)):
        for s in inputList:
            if s == featureVec[i]: 
                newVec[i] += 1

    return newVec.tolist()

############################################################

def learnPredictor(trainExamples, countryVec, sectorVec, issueVec):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (country, sector, issue)
    tuples), a |featureExtractor| to apply to x, and the number of iterations to train 
    |numIters|, the step size |eta|, return the matrix learned.

    Implements stochastic gradient descent.
    '''
    numIters = 20
    eta = 0.01

    numTrainers = len(trainExamples)
    numFeatures = len(countryVec) + len(sectorVec)
    numIssues = len(issueVec)
    weights=np.zeros((numFeatures, numIssues))

    for i in range(numIters):
        for j in range(numTrainers):

            x = featurize(trainExamples[j][0], countryVec)+featurize(trainExamples[j][1], sectorVec)
            y = featurize(trainExamples[j][2], issueVec)

            #Check every issue's feature vector in the weights matrix
            for k in range(numIssues):
                regression = (np.dot(x, weights[:,k]) - y[k])**2

                if regression >1: #IDK what it's supposed to be less than
                    weights[:,k] = weights[:,k] + np.multiply(eta*y[k], x)

    return weights

############################################################

df = prepare_raw_data()
countries = get_unique(df['Country'])
sectors = get_unique(df['Sector/Industry (1)'].append(df['Sector/Industry (2)']))
issues = get_unique(df['Issue Raised (1)'].append(df['Issue Raised (2)']).append(df['Issue Raised (3)']).append(df['Issue Raised (4)']).append(df['Issue Raised (5)']).append(df['Issue Raised (6)']).append(df['Issue Raised (7)']).append(df['Issue Raised (8)']).append(df['Issue Raised (9)']).append(df['Issue Raised (10)']))
clean_df = prepare_clean_data(df)

weights = learnPredictor(clean_df[:600], countries, sectors, issues)
print weights

#################################################################
def testTests(testExamples, countryVec, sectorVec, issueVec, weights):
    numTests = len(testExamples)
    numFeatures = len(countryVec) + len(sectorVec)
    numIssues = len(issueVec)
    weights=np.zeros((numFeatures, numIssues))

    numSuccesses = 0

    for j in range(numTests):

        x = featurize(testExamples[j][0], countryVec).append(featurize(testExamples[j][1], sectorVec))
        y = featurize(testExamples[j][2], issueVec)

        #Check every issue's feature vector in the weights matrix
        y_predicted = np.matmul(weights, x)
        guesses = get_max_guess(y_predicted, issueVec)
        count = 0
        for guess in guesses:
            if guess in testExamples[j][2]:
                count+=1
        print '# of correct guesses = %d out of 3' % count
        if count > 0: numSuccesses+=1

    # END_YOUR_CODE
    return weights 

def get_max_guess(guesses, issueVec):
    guesses = [] 
    for x in range(3):
        index_max = max(xrange(len(guesses)), key=y.__getitem__)
        guesses.append(issueVec[index_max])
        del guesses[max_index]
    return guesses
        
print('Test')
testTests(clean_df[600:], countries, sectors, issues, weights)
