import pandas as py

def get_project_names(): 
    ac = pd.read_csv('2016_17_complaints.csv')
    return list(set(filter(None, list(ac['Project Name'].fillna('')))))
    # to use: 
    # test = encoded[2] where "'Rocade Ext\xc3\xa9rieure du Grand Tunis'"
    # print(test == 'Rocade Extérieure du Grand Tunis')
