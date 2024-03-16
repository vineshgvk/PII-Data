'''
resampling.py
Downsampling of negative examples
'''
import json
import numpy as np
import os
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

def process_data_and_save(input_file_path):
    try:
        # Load data from a JSON file
        with open(input_file_path, "r") as json_file:
            data = json.load(json_file)

        # Downsampling of negative examples
        p = []  # positive samples (contain relevant labels)
        n = []  # negative samples (presumably contain entities that are possibly wrongly classified as entity)
        for d in data:
            if any(label != "O" for label in d["labels"]):
                p.append(d)
            else:
                n.append(d)

        # Combine data
        data_n = data + p + n[:len(n) // 3]

        # Specify the output file path
        output_file_path = os.path.join(PROJECT_DIR,"data","processed","resampled.json")

        # Save the processed data
        with open(output_file_path, "w") as output_file:
            json.dump(data_n, output_file)

        print("Processed data saved successfully.")

    except Exception as e:
        logging.critical('Failure! resampling.py failed!')
        print(f"An unexpected error occurred: {e}")

# Path to the JSON file
input_file_path = os.path.join(PROJECT_DIR,"data","processed","train.json")

# Call the function to process data and save it
process_data_and_save(input_file_path)