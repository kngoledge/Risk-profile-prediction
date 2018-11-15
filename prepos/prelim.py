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
        [countries], [sectors], [issues]) for every datapoint
    """
    clean_data = [] 
    for index, row in df.iterrows():
        clean_sectors = filter(None,x['Sector/Industry (1)'].split('|')+x['Sector/Industry (2)'].split('|'))
        clean_issues = filter(None,x['Issue Raised (1)'].split('|')+x['Issue Raised (2)'].split('|')+x['Issue Raised (3)'].split('|')+x['Issue Raised (4)'].split('|')+x['Issue Raised (5)'].split('|')+x['Issue Raised (6)'].split('|')+x['Issue Raised (7)'].split('|')+x['Issue Raised (8)'].split('|')+x['Issue Raised (9)'].split('|')+x['Issue Raised (10)'].split('|'))
        clean_tuple = (row['Country'].split('|'), clean_sectors, clean_issues)
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

    
df = prepare_data()
country = get_unique(df['Country'])
sector = get_unique(df['Sector/Industry (1)'].append(df['Sector/Industry (2)']))
issues = get_unique(df['Issue Raised (1)'].append(df['Issue Raised (2)']).append(df['Issue Raised (3)']).append(df['Issue Raised (4)']).append(df['Issue Raised (5)']).append(df['Issue Raised (6)']).append(df['Issue Raised (7)']).append(df['Issue Raised (8)']).append(df['Issue Raised (9)']).append(df['Issue Raised (10)']))
clean_df = prepare_clean_data(df)

