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

sum_x0test = np.zeros(featureVec_size)
sum_y0test = np.zeros(numIssues)
sum_y0pred = np.zeros(numIssues)
sum_x1test = np.zeros(featureVec_size)
sum_y1test = np.zeros(numIssues)
sum_y1pred = np.zeros(numIssues)
sum_x2test = np.zeros(featureVec_size)
sum_y2test = np.zeros(numIssues)
sum_y2pred = np.zeros(numIssues)
sum_x3test = np.zeros(featureVec_size)
sum_y3test = np.zeros(numIssues)
sum_y3pred = np.zeros(numIssues)
sum_x4test = np.zeros(featureVec_size)
sum_y4test = np.zeros(numIssues)
sum_y4pred = np.zeros(numIssues)
sum_x5test = np.zeros(featureVec_size)
sum_y5test = np.zeros(numIssues)
sum_y5pred = np.zeros(numIssues)
sum_x6test = np.zeros(featureVec_size)
sum_y6test = np.zeros(numIssues)
sum_y6pred = np.zeros(numIssues)

for i in range(2500):
  if ytest[i][-1] != 1.:
    #print ('\n TEST', i, ': ', xtest[i], ytest[i], ypred[i])


    if ytest[i][0] == 1:
      sum_x0test += xtest[i]
      sum_y0test += ytest[i]
      sum_y0pred += ypred[i]

    if ytest[i][1] == 1:
      sum_x1test += xtest[i]
      sum_y1test += ytest[i]
      sum_y1pred += ypred[i]

    if ytest[i][2] == 1:
      sum_x2test += xtest[i]
      sum_y2test += ytest[i]
      sum_y2pred += ypred[i]

    if ytest[i][3] == 1:
      sum_x3test += xtest[i]
      sum_y3test += ytest[i]
      sum_y3pred += ypred[i]

    if ytest[i][4] == 1:
      sum_x4test += xtest[i]
      sum_y4test += ytest[i]
      sum_y4pred += ypred[i]

    if ytest[i][5] == 1:
      sum_x5test += xtest[i]
      sum_y5test += ytest[i]
      sum_y5pred += ypred[i]

    if ytest[i][6] == 1:
      sum_x6test += xtest[i]
      sum_y6test += ytest[i]
      sum_y6pred += ypred[i]

#['Community', 'Damages', 'Displacement', 'Environment', 'Malpractice', 'Other', 'Violence', 'NONE']

print "BASED ON EXPECTED"
print('COMMUNITY')
print('sum_xtest = ', sum_x0test)
print('sum_ytest = ', sum_y0test)
print('sum_ypred = ', sum_y0pred)

print('DAMAGES')
print('sum_xtest = ', sum_x1test)
print('sum_ytest = ', sum_y1test)
print('sum_ypred = ', sum_y1pred)

print('DISPLACEMENT')
print('sum_xtest = ', sum_x2test)
print('sum_ytest = ', sum_y2test)
print('sum_ypred = ', sum_y2pred)

print('ENVIRONMENT')
print('sum_xtest = ', sum_x3test)
print('sum_ytest = ', sum_y3test)
print('sum_ypred = ', sum_y3pred)

print('MALPRACTICE')
print('sum_xtest = ', sum_x4test)
print('sum_ytest = ', sum_y4test)
print('sum_ypred = ', sum_y4pred)

print('OTHER')
print('sum_xtest = ', sum_x5test)
print('sum_ytest = ', sum_y5test)
print('sum_ypred = ', sum_y5pred)

print('VIOLENCE')
print('sum_xtest = ', sum_x6test)
print('sum_ytest = ', sum_y6test)
print('sum_ypred = ', sum_y6pred)


sum_x0test = np.zeros(featureVec_size)
sum_y0test = np.zeros(numIssues)
sum_y0pred = np.zeros(numIssues)
sum_x1test = np.zeros(featureVec_size)
sum_y1test = np.zeros(numIssues)
sum_y1pred = np.zeros(numIssues)
sum_x2test = np.zeros(featureVec_size)
sum_y2test = np.zeros(numIssues)
sum_y2pred = np.zeros(numIssues)
sum_x3test = np.zeros(featureVec_size)
sum_y3test = np.zeros(numIssues)
sum_y3pred = np.zeros(numIssues)
sum_x4test = np.zeros(featureVec_size)
sum_y4test = np.zeros(numIssues)
sum_y4pred = np.zeros(numIssues)
sum_x5test = np.zeros(featureVec_size)
sum_y5test = np.zeros(numIssues)
sum_y5pred = np.zeros(numIssues)
sum_x6test = np.zeros(featureVec_size)
sum_y6test = np.zeros(numIssues)
sum_y6pred = np.zeros(numIssues)

prob = 0.1

for i in range(2500):
  if ytest[i][-1] != 1.:
    #print ('\n TEST', i, ': ', xtest[i], ytest[i], ypred[i])


    if ypred[i][0] > prob:
      sum_x0test += xtest[i]
      sum_y0test += ytest[i]
      sum_y0pred += ypred[i]

    if ypred[i][1] > prob:
      sum_x1test += xtest[i]
      sum_y1test += ytest[i]
      sum_y1pred += ypred[i]

    if ypred[i][2] > prob:
      sum_x2test += xtest[i]
      sum_y2test += ytest[i]
      sum_y2pred += ypred[i]

    if ypred[i][3] > prob:
      sum_x3test += xtest[i]
      sum_y3test += ytest[i]
      sum_y3pred += ypred[i]

    if ypred[i][4] > prob:
      sum_x4test += xtest[i]
      sum_y4test += ytest[i]
      sum_y4pred += ypred[i]

    if ypred[i][5] > prob:
      sum_x5test += xtest[i]
      sum_y5test += ytest[i]
      sum_y5pred += ypred[i]

    if ypred[i][6] > prob:
      sum_x6test += xtest[i]
      sum_y6test += ytest[i]
      sum_y6pred += ypred[i]

#['Community', 'Damages', 'Displacement', 'Environment', 'Malpractice', 'Other', 'Violence', 'NONE']

print "BASED ON PREDICTIONS"
print('COMMUNITY')
print('sum_xtest = ', sum_x0test)
print('sum_ytest = ', sum_y0test)
print('sum_ypred = ', sum_y0pred)

print('DAMAGES')
print('sum_xtest = ', sum_x1test)
print('sum_ytest = ', sum_y1test)
print('sum_ypred = ', sum_y1pred)

print('DISPLACEMENT')
print('sum_xtest = ', sum_x2test)
print('sum_ytest = ', sum_y2test)
print('sum_ypred = ', sum_y2pred)

print('ENVIRONMENT')
print('sum_xtest = ', sum_x3test)
print('sum_ytest = ', sum_y3test)
print('sum_ypred = ', sum_y3pred)

print('MALPRACTICE')
print('sum_xtest = ', sum_x4test)
print('sum_ytest = ', sum_y4test)
print('sum_ypred = ', sum_y4pred)

print('OTHER')
print('sum_xtest = ', sum_x5test)
print('sum_ytest = ', sum_y5test)
print('sum_ypred = ', sum_y5pred)

print('VIOLENCE')
print('sum_xtest = ', sum_x6test)
print('sum_ytest = ', sum_y6test)
print('sum_ypred = ', sum_y6pred)


# Results
print "\nCompiled! Here are your results..."
print('Test loss:', score[0])
print('Test accuracy:', score[1])
