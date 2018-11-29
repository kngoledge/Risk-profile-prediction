import os, random, operator, sys
import collections
import math
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import keras.backend as K

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

def calc_acc(y_true, y_pred):
    numIssues = np.sum(y_true)
    count = 0.0
    for i in len(y_true):
        if y_true[i] == 1 and y_pred[i] > 0.5: count+=1.0

    return count/numIssues

############################################################
# Clean the dataset 
############################################################

df = prepare_raw_data()
countries = get_unique(df['Country'])
sectors = get_unique(df['Sector/Industry (1)'].append(df['Sector/Industry (2)']))
sectors.pop(0)
issues = get_unique(df['Issue Raised (1)'].append(df['Issue Raised (2)']).append(df['Issue Raised (3)']).append(df['Issue Raised (4)']).append(df['Issue Raised (5)']).append(df['Issue Raised (6)']).append(df['Issue Raised (7)']).append(df['Issue Raised (8)']).append(df['Issue Raised (9)']).append(df['Issue Raised (10)']))
issues.pop(0)
clean_df = prepare_clean_data(df)

xtrain = []
ytrain = []
numTrainers = 600       #change to 600
trainExamples = clean_df[:numTrainers]
for i in range(numTrainers):
    x = featurize(trainExamples[i][0], countries)+featurize(trainExamples[i][1], sectors)
    y = featurize(trainExamples[i][2], issues)
    xtrain.append(x)
    ytrain.append(y)
xtrain = np.asarray(xtrain)
ytrain = np.asarray(ytrain)
print("shape of x_train: ", xtrain.shape)
print("y train: ", ytrain.shape)

xtest = []
ytest = []
testExamples = clean_df[numTrainers:]
numTesters = len(testExamples)      #change to len(testExamples)
for i in range(numTesters):
    x = featurize(trainExamples[i][0], countries)+featurize(trainExamples[i][1], sectors)
    y = featurize(trainExamples[i][2], issues)
    xtest.append(x)
    ytest.append(y)
xtest = np.asarray(xtest)
ytest = np.asarray(ytest)

# Parameters
featureVec_size = len(countries) + len(sectors) 
final_dim = len(issues)
batch_sz = 100

print ("Feature Vect Size: ", featureVec_size)
print ("Issues: ", final_dim)
print ("Num Countries: ", len(countries))
print countries
print sectors
print issues

############################################################

# Create the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(featureVec_size,) ))
model.add(Dense(64, activation='relu'))
model.add(Dense(final_dim, activation='softmax'))

model.summary()

# Stochastic Gradient Descent
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(xtrain, ytrain, epochs=50, batch_size=batch_sz)

# Time to test!
score = model.evaluate(xtest, ytest, batch_size=batch_sz)

print "compiled!"
print('Test loss:', score[0])
print('Test accuracy:', score[1])




