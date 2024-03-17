"""
Load data from the input JSON file, validate datatypes and formats.

:param input_json_path: Path to the input JSON file.

"""

import os
import json
import logging
import pandas as pd

# Stash the logs in the data/logs path.
logsPath = os.path.abspath(os.path.join(os.getcwd(), 'data', 'logs'))
if not os.path.exists(logsPath):
    # Create the folder if it doesn't exist
    os.makedirs(logsPath)
    print(f"Folder '{logsPath}' created successfully.")

logging.basicConfig(filename=os.path.join(logsPath, 'logs.log'),  # log filename with today's date.
                    filemode="w",  # write mode
                    level=logging.ERROR,  # Set error as the default log level.
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # logging format
                    datefmt='%Y-%m-%d %H:%M:%S', )  # logging (asctime) date format

# Get the current working directory
PROJECT_DIR = os.getcwd()
# jsonPath is input to function. outPklPath is path after processing.
jsonPath = os.path.join(PROJECT_DIR, 'data', 'processed', 'train.json')

textThreshold = 25  # Remove records with full_text < 25 words.
trainSamples = 100  # Needs at least trainSamples amount of records for training.
expectedDtypes = {'document': int,
                  'full_text': object,
                  'tokens': object,
                  'trailing_whitespace': object,
                  'labels': object
                  }


def anomalyDetect(inputPath=jsonPath,
                  textThreshold=textThreshold,
                  trainSamples=trainSamples,
                  expectedDtypes=expectedDtypes):
    '''
    anomalyDetect looks for the right data types, and checks for text length below a threshold
    Args:
        inputPath: Input JSON path after process_data_and_save.
        outputPath: Output pickle path after dupeRemoval processing.
    Returns:
        outputPath
    '''

    # Open file in read mode if exists
    if os.path.exists(inputPath):
        with open(inputPath, "r") as file:
            data = json.load(file)
    else:
        raise FileNotFoundError(f"FAILED! No such path at {inputPath}")

    # Convert JSON data to DataFrame
    df = pd.DataFrame(data)

    # Check for text length
    rowsRemoved = 0
    for index, row in df.iterrows():
        if len(row['full_text'].split()) < textThreshold:
            rowsRemoved += 1
            df.drop(index, inplace=True)
    print(f'Records removed because of text length threshold {textThreshold}: {rowsRemoved} records')

    # Check for trainSamples threshold for training
    if df.shape[0] < trainSamples:
        print(f'Not enough training samples for model to be trained')

    # Check for appropriate text types
    for col in df.columns:
        if df[col].dtype != expectedDtypes[col]:
            print(f'{col} data type mismatch')
            print(df[col].dtype)

    # Check for tokens length to be >25
    for index, row in df.iterrows():
        if len(row['tokens']) < 25:
            logging.error(f"Tokens size less than 25 in row {index}")

    # Check if trailing_whitespace is int and has only values 1 or 0
    valid_values = [True,False]
    if 'trailing_whitespace' in df.columns:
        if not df['trailing_whitespace'].isin(valid_values).all():
            logging.error("The 'trailing_whitespace' column contains values other than 1 or 0.")
            print("The 'trailing_whitespace' column contains values other than 1 or 0.")
    else:
        logging.error("The 'trailing_whitespace' column is missing.")
        print("The 'trailing_whitespace' column is missing.")

    # Check if labels are one of the 12 unique values
    allowed_labels = ['B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS',
                      'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM',
                      'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O']

    if 'labels' in df.columns:
        # Flatten the list of labels
        flat_labels = df['labels'].explode()
        
        # Check for invalid labels
        invalid_labels = flat_labels[~flat_labels.isin(allowed_labels)]
        if not invalid_labels.empty:
            logging.error(f"The 'labels' column contains invalid values: {invalid_labels.unique()}.")
    else:
        logging.error("The 'labels' column is missing.")
    return True


anomalyDetect(inputPath=jsonPath, textThreshold=textThreshold, trainSamples=trainSamples,expectedDtypes=expectedDtypes)
