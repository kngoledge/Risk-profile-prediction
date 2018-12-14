import os, random, operator, sys
import collections
import math
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import keras.backend as K
import util
from sklearn.model_selection import StratifiedKFold

############################################################

# cross validation function
# PARAMETERS:
# feat_vecs: list of feature vector arrays from our labeled data
# labels: list of output arrays for those feature vectors
# n_folds: number of times we want to perform cross validation
# RETURNS: tuple of result metrics

def perform_validation(model, feat_vecs, labels, all_labels, n_folds):
	skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=False)
	min_loss, max_loss = float('inf'), -float('inf')
	min_accuracy, max_accuracy = float('inf'), -float('inf')
	loss_sum = 0
	accuracy_sum = 0
	# iterate through n_folds number of train/test partitions
	for train_indices, test_indices in skf.split(feat_vecs, labels):
		sz = len(train_indices)
		xtrain, ytrain, xtest, ytest = [], [], [], []
		for i, train_index in enumerate(train_indices):
			xtrain.append(feat_vecs[train_index])
			ytrain.append(all_labels[train_index])
		for i, test_index in enumerate(test_indices):
			xtest.append(feat_vecs[test_index])
			ytest.append(all_labels[test_index])
		# train on xtrain, ytrain (copied from neuralnet.py)
		xtrain = np.asarray(xtrain)
		ytrain = np.asarray(ytrain)
		xtest = np.asarray(xtest)
		ytest = np.asarray(ytest)
		history = model.fit(xtrain, ytrain, epochs=200, batch_size=batch_sz, verbose=0)
		# test on xtest, ytest (copied from neuralnet.py)
		score = model.evaluate(xtest, ytest, batch_size=batch_sz)
		# update metrics
		loss_sum += score[0]
		accuracy_sum += score[1]
		if score[0] < min_loss: min_loss = score[0]
		if score[0] > max_loss: max_loss = score[0]
		if score[1] < min_accuracy: min_accuracy = score[1]
		if score[1] > max_accuracy: max_accuracy = score[1]

	avg_loss = float(loss_sum)/n_folds
	avg_accuracy = float(accuracy_sum)/n_folds
	final_tuple = (avg_loss, min_loss, max_loss, avg_accuracy, min_accuracy, max_accuracy)
	print ("average loss: %f\nmin: %f\tmax: %f\n\naverage accuracy: %f\nmin: %f\tmax: %f\t\n" % final_tuple)
	return final_tuple

def cross_validate(model, feat_vecs, labels, n_folds):
	summaries = []
	for col in range(len(labels[0])):
		print col
		issue_label = []
		for label in labels:
			issue_label.append(label[col])
		summaries.append(perform_validation(model, feat_vecs, issue_label, labels, n_folds))
	return summaries

def evaluate_summaries(summaries):
	max_tuple = []
	avg_tuple = []
	for i, summary in enumerate(summaries):
		if len(max_tuple) == 0 or max_tuple[5] < summary[5]:
			max_tuple = [summary, i]
		if len(avg_tuple) == 0 or avg_tuple[3] < summary[3]:
			avg_tuple = [summary, i]
	print "Out of the 24 iterations, issue #%d had the highest max accuracy at %f, \n\
	and issue #%d had the highest average accuracy at %f." % (max_tuple[1], max_tuple[0], avg_tuple[1], avg_tuple[0])

############################################################

# Parameters
numTrainers = 6000
xlist, ylist, numRegions, numSectors, numIssues, numMoney = util.organize_data()
featureVec_size = numRegions + numSectors + numMoney
final_dim = numIssues
batch_sz = 256

xtrain = np.array( xlist[:numTrainers] )
ytrain = np.array( ylist[:numTrainers] )
xtest = np.array( xlist[numTrainers:] )
ytest = np.array( ylist[numTrainers:] )

# Create the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(featureVec_size,) ))	#small dataset - less hidden layers needed
model.add(Dense(40, activation='relu'))
model.add(Dense(final_dim, activation='sigmoid'))

model.summary()

# Stochastic Gradient Descent
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy',
			  optimizer=sgd,
			  metrics=['accuracy'])

history = model.fit(xtrain, ytrain, epochs=200, batch_size=batch_sz)

# Time to test!
score = model.evaluate(xtest, ytest, batch_size=batch_sz)

print "\nCompiled! Here are your results..."
print('Test loss:', score[0])
print('Test accuracy:', score[1])

all_featurevecs = xlist
all_labels = ylist
summaries = cross_validate(model, all_featurevecs, all_labels, 3)
evaluate_summaries(summaries)

############################################################
"""
	NOTES: 
	- Consider shrinking the number of issues (from 15 to 4) - leads to acc of 100%... too high

"""