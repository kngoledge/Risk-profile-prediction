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


############################################################

# Parameters
numTrainers = 5000
xtrain, ytrain, xtest, ytest, numRegions, numSectors, numIssues = util.organize_data(numTrainers)
featureVec_size = numRegions + numSectors
final_dim = numIssues
batch_sz = 256

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

############################################################
"""
	NOTES: 
	- Consider shrinking the number of issues (from 15 to 4) - leads to acc of 100%... too high

"""