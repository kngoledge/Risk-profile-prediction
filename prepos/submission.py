#!/usr/bin/python

import random
import collections
import math
import sys
import numpy as np
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    d = {}
    for w in x.split():
        if w not in d: 
            d[w]=0
        d[w]+=1
    return d
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    weights={}
    numTrainers = len(trainExamples)
    for i in range(numIters):
        for j in range(numTrainers):
            featureVector = featureExtractor(trainExamples[j][0])
            y = trainExamples[j][1]
            margin = dotProduct(weights, featureVector)*y
            if margin < 1:
                increment(weights,eta*y,featureVector)
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = {}
        dicSize = len(weights)
        for i in xrange(dicSize):
            # get rand int for index 
            index = random.randrange(0, dicSize)
            phi[weights.keys()[index]] = random.randrange(1,20)

        if dotProduct(phi, weights)>=0:
            y = 1
        else: 
            y = -1

        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        # print x
        x = x.replace(" ","")        
        dic = {}
        for i in range(len(x)-n+1):
            subword = x[i:i+n]
            if subword not in dic:
                dic[subword] = 0
            dic[subword] += 1

        return dic

        # END_YOUR_CODE
    return extract


############################################################
		#NEW STUFF
############################################################


############################################################

def featurize(sector, country, sectorVec, countryVec):
    """
    Converts string inputs (sector and country) into an
    extracted feature vector, based on the related feature vectors.
    Outputs a sparse feature vector that is the concatenation of
    the sector feature vec followed by the country feature vec. 
    """
    
    newVec = np.zeros(len(sectorVec)+len(countryVec))

    for i in range(len(sectorVec)):
        if sector == sectorVec[i]: 
            newVec[i] += 1
	for j in range(len(countryVec)):
		if country == countryVec[j]:
			newVec[j+len(sectorVec)] += 1

    return newVec

############################################################
# New Code

def learnPredictor(countryVec, sectorVec, issueVec, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    weights={}
    numTrainers = len(trainExamples)
    for i in range(numIters):
        for j in range(numTrainers):
            featureVector = featureExtractor(trainExamples[j][0])
            y = trainExamples[j][1]
            margin = dotProduct(weights, featureVector)*y
            if margin < 1:
                increment(weights,eta*y,featureVector)
    # END_YOUR_CODE
    return weights

############################################################


