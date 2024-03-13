'''
duplicates.py
Checks for duplicates in the dataset and removes any.
'''
import os
import pickle
import logging

# Get the current working directory
PROJECT_DIR = os.getcwd()
# pklPath is input to function. outPklPath is path after processing.

pklPath = os.path.join(PROJECT_DIR,"data","processed",'missing_values.pkl')
outPklPath = os.path.join(PROJECT_DIR, "data","processed",'dup_removed.pkl')

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

def dupeRemoval(inputPath=pklPath, outputPath=outPklPath):
    '''
    dupeRemoval checks for duplicates in the dataset and removes any.
    Args:
        inputPath: Input pickle path after loadData.
        outputPath: Output pickle path after dupeRemoval processing.
    Returns:
        outputPath
    '''

    print("Loading data from:", inputPath)
    # Open file in read-binary mode if exists
    if os.path.exists(inputPath):
        with open(inputPath, "rb") as file:
            df = pickle.load(file)
    else:
        logging.critical('FAILURE! duplicates.py error!')
        raise FileNotFoundError(f"FAILED! No such path at {inputPath}")
    
    print("Original DataFrame:")
    print(df.head())

    # Removes rows i.e. any duplicates in the full_text column
    df.drop_duplicates(subset=['full_text'], inplace = True)

    print("DataFrame after removing duplicates:")
    print(df.head())

    # Pickle dataset and push forward
    # opening file in write-binary mode
    with open(outputPath, "wb") as file:
        pickle.dump(df, file)
    print(f'Data pickled after dupeRemoval at {outputPath}')

    return outputPath

dupeRemoval(inputPath=pklPath, outputPath=outPklPath)