import os, random, operator, sys
import collections
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset 

df = prepare_raw_data()
countries = get_unique(df['Country'])
sectors = get_unique(df['Sector/Industry (1)'].append(df['Sector/Industry (2)']))
sectors.pop(0)
issues = get_unique(df['Issue Raised (1)'].append(df['Issue Raised (2)']).append(df['Issue Raised (3)']).append(df['Issue Raised (4)']).append(df['Issue Raised (5)']).append(df['Issue Raised (6)']).append(df['Issue Raised (7)']).append(df['Issue Raised (8)']).append(df['Issue Raised (9)']).append(df['Issue Raised (10)']))
issues.pop(0)
clean_df = prepare_clean_data(df)

xtrain_dataset = []
ytrain_dataset = []

numTrainers = 600
trainExamples = clean_df[:numTrainers]
for i in range(numTrainers):
    x = featurize(trainExamples[i][0], countries)+featurize(trainExamples[i][1], sectors)
    y = featurize(trainExamples[i][2], issues)
    xtrain_dataset.append(x)
    ytrain_dataset.append(y)

xtest_dataset = []
ytest_dataset = []
testExamples = clean_df[numTrainers:]
for i in range(len(testExamples)):
    x = featurize(trainExamples[i][0], countries)+featurize(trainExamples[i][1], sectors)
    y = featurize(trainExamples[i][2], issues)
    xtest_dataset.append(x)
    ytest_dataset.append(y)

# Hyper-parameters 
input_size = len(countries) + len(sectors)
hidden_size = 100
num_classes = 10
num_epochs = len(issues)
batch_size = 100
learning_rate = 0.001

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model - SGD 
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i in len(train_loader):  
        # Move tensors to the configured device
        xtrain_dataset 
        ytrain_dataset 
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')