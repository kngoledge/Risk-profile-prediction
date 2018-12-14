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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt


############################################################
# Constructs a neural network 
# Imports util, namely feature vectors of input and output
# Provides predicted output
############################################################

### Parameters
numTrainers = 6000	# DECIDE SPLIT BETWEEN TRAINING AND TEST
xlist, ylist, numRegions, numSectors, numIssues, numMoney = util.organize_data()
featureVec_size = numRegions + numSectors + numMoney
final_dim = numIssues
batch_sz = 256

xtrain = np.array( xlist[:numTrainers] )
ytrain = np.array( ylist[:numTrainers] )
xtest = np.array( xlist[numTrainers:] )
ytest = np.array( ylist[numTrainers:] )

### Create the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(featureVec_size,) ))	#small dataset - less hidden layers needed
model.add(Dense(50, activation='relu'))
model.add(Dense(final_dim, activation='sigmoid'))

model.summary()

### Stochastic Gradient Descent
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(xtrain, ytrain, epochs=200, batch_size=batch_sz)

### Time to test!
score = model.evaluate(xtest, ytest, batch_size=batch_sz)
ypred = model.predict(xtest)

### Prints out project input/output pairs and their predictions.
### Uncomment for more information.
# num_predicted_issues = 0
# for i in range(len(ytest)):
#   if ytest[i][-1] != 1.:
#     print ('\n TEST', i, ': ', xtest[i], ytest[i], ypred[i])
#     num_predicted_issues += 1
# print('Number of projects predicted to have issues: ', num_predicted_issues)

### Results
print "\nCompiled! Here are your results..."
print('Test loss:', score[0])
print('Test accuracy:', score[1])
