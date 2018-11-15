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
    Given |trainExamples| (each element is a list of (country, sector, issue)
    tuples), the relevant feature vectors, return the matrix learned.

    Implements stochastic gradient descent.
    '''
    numIters = 20   # Can pass this in to constructor later
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
                residual = (np.dot(x, weights[:,k]) - y[k])**2

                if residual > 10: #IDK what it's supposed to be less than
                    weights[:,k] = weights[:,k] + np.multiply(eta*y[k], x)

    return weights



############################################################

def predictOutput(testExamples, countryVec, sectorVec, issueVec, weights):
    """ 
    Given |testExamples| (each element is a list of (country, sector, issue)
    tuples), the relevant feature vectors, and the weight matrix, returns a
    vector of the predicted outputs. 
    """

    numTests = len(testExamples)
    numFeatures = len(countryVec) + len(sectorVec)
    numIssues = len(issueVec)
    weights= np.zeros((numFeatures, numIssues))
    y_predicted = []

    for j in range(numTests):

        x = featurize(testExamples[j][0], countryVec) + featurize(testExamples[j][1], sectorVec)
        y = featurize(testExamples[j][2], issueVec)

        #Check every issue's feature vector in the weights matrix
        y_predicted.append((np.matmul(np.asarray(x), weights)).tolist())

    return y_predicted 

############################################################

def checkAccuracy(y_predicted, y_actual, issueVec):
    """ 
    Compares the vector |y_predicted| to the vector |y_actual|, 
    when using complain data as part of test set. 
    Returns rough estimate of the % of successful predictions. 
    """
    
    numSuccesses = 0
    for i in range(len(y_predicted)):
        predictions = getMaxGuess(y_predicted[i], issueVec)
        count = 0
        for prediction in predictions:
            if prediction in y_actual[i]:
                count+=1
        print '# of correct guesses = %d out of 3' % count
        if count > 0: numSuccesses+=1
    return numSuccesses/(len(y_predicted)*3.0) 

############################################################

def getMaxGuess(guesses, issueVec):
    """
    Converts featurized vector guess into a list of the three
    most probable issues. Returns a list of issues.
    """
    new_guesses = [] 
    for x in range(3):
        index_max = max(xrange(len(guesses)), key=guesses.__getitem__)
        guesses.append(issueVec[index_max])
        del guesses[max_index]
    return guesses
        
############################################################
""" 
    Implements all of our functions. 
"""

df = prepare_raw_data()
#np.random.shuffle(df)
countries = get_unique(df['Country'])
sectors = get_unique(df['Sector/Industry (1)'].append(df['Sector/Industry (2)']))
issues = get_unique(df['Issue Raised (1)'].append(df['Issue Raised (2)']).append(df['Issue Raised (3)']).append(df['Issue Raised (4)']).append(df['Issue Raised (5)']).append(df['Issue Raised (6)']).append(df['Issue Raised (7)']).append(df['Issue Raised (8)']).append(df['Issue Raised (9)']).append(df['Issue Raised (10)']))
clean_df = prepare_clean_data(df)

weights = learnPredictor(clean_df[:600], countries, sectors, issues)
print weights
predictedOutputs = predictOutput(clean_df[600:], countries, sectors, issues, weights)
acc = checkAccuracy(predictedOutputs, clean_df[600:][2],issues)
print('Test')
testTests(clean_df[600:], countries, sectors, issues, weights)
