'''
loadData
Loads json files after unzipping.
Step 1 in data processing post downloading.
'''
import pandas as pd
import pickle
import os
import logging

rootDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
jsonPath = os.path.join(rootDir, 'data', 'train.json')
pklPath = os.path.join(rootDir, 'data', 'processed', 'rawData.pkl')

def loadDataset(jsonPath = jsonPath,
                pklPath = pklPath): # Returns pickled dataset path
    '''
    loadDataset to load dataset from the said jsonPath.
    Args:
        jsonPath: JSON file path.
    Returns:
        pklPath: Returns file path to the loaded path. 
    '''
    # If JSON file exists
    if os.path.exists(jsonPath):
        df = pd.read_json(jsonPath)
        print(f'SUCCESS! Data loaded from {jsonPath}')
    else:
        print(f'FAILED! No data at the specified location {jsonPath}')
        raise FileNotFoundError(f'FAILED! No data at the specified location {jsonPath}')
    
    # Pickling the dataset in the pklPath
    os.makedirs(os.path.dirname(pklPath), 
                exist_ok = True) # exist_ok = If the specified directory already exists and value is set to False an OSError is raised, else not.
    
    # Opening 'file' in write-binary mode
    with open(pklPath, "wb") as file:
        pickle.dump(df, file)
    print(f"Data saved at {pklPath}")
    
    return pklPath

# loadDataset(jsonPath = jsonPath, pklPath = pklPath)