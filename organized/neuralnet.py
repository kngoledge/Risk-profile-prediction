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


# first, add threshold to calculate 0/1 values
def change_by_threshold(threshold, values_vector):
  new_values_vector = [] 
  for x in values_vector:
    actual = [] 
    for y in x: 
      y = 1 if y > threshold else 0 
      actual.append(y)
    new_values_vector.append(actual)
  return new_values_vector


############################################################

# Parameters
numTrainers = 6000	# DECIDE SPLIT BETWEEN TRAINING AND TEST
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
model.add(Dense(50, activation='relu'))
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
ypred = model.predict(xtest)

for i in range(1000):
  if ytest[i][-1] != 1.:
    print ('\n TEST', i, ': ', xtest[i], ytest[i], ypred[i])

"""
# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(23):
    precision[i], recall[i], _ = precision_recall_curve(ytest[:, i],
                                                        ypred[:, i])
    average_precision[i] = average_precision_score(ytest[:, i], ypred[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(ytest.ravel(),
    ypred.ravel())
average_precision["micro"] = average_precision_score(ytest, ypred,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

## TO DO: get an average for micro recall 
"""

print "\nCompiled! Here are your results..."
print('Test loss:', score[0])
print('Test accuracy:', score[1])

############################################################
"""
	NOTES: 
	- Consider shrinking the number of issues (from 15 to 4) - leads to acc of 100%... too high

"""