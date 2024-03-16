'''
missing_values.py
Looks for N/As in the dataset and removes relevent records accordingly.
data_download > missing_values (returns missing_values.pkl)
'''
import pandas as pd
import os
import pickle
import json
import logging

# Stash the logs in the data/logs path.
logsPath = os.path.abspath(os.path.join(os.getcwd(), 'data', 'logs'))
if not os.path.exists(logsPath):
    # Create the folder if it doesn't exist
    os.makedirs(logsPath)
    print(f"Folder '{logsPath}' created successfully.")

logging.basicConfig(filename = os.path.join(logsPath, 'logs.log'), # log filename with today's date.
                    filemode = "w", # write mode
                    level = logging.ERROR, # Set error as the default log level.
                    format ='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # logging format
                    datefmt = '%Y-%m-%d %H:%M:%S',) # logging (asctime) date format

# Get the current working directory
PROJECT_DIR = os.getcwd()
# jsonPath is input to function. outPklPath is path after processing.
jsonPath = os.path.join(PROJECT_DIR, "data", "processed", "train.json")
outPklPath = os.path.join(PROJECT_DIR, "data", "processed", "missing_values.pkl")

def naHandler(inputPath=jsonPath, outputPath=outPklPath):
    if os.path.exists(inputPath):
        print("Loading data from:", inputPath)
        with open(inputPath, "r") as file:
            data = json.load(file)
            df = pd.DataFrame(data)
    else:
        raise FileNotFoundError(f"FAILED! No such path at {inputPath}")

    # Remove NAs wherever applicable.
    df.dropna(subset=['full_text'], inplace=True)

    nullCount = df.isnull().sum().sum()
    if nullCount > 0:
        nullsPresentError = f'Nulls {nullCount} still present in the dataset'
        logging.critical('FAILURE! missing_values.py error!')
        print(nullsPresentError)
        raise ValueError(nullsPresentError)

    # Pickle dataset and push forward
    with open(outputPath, "wb") as file:
        pickle.dump(df, file)
    print(f'Data pickled after naHandling at {outputPath}')
    return outputPath

naHandler(inputPath=jsonPath, outputPath=outPklPath)
