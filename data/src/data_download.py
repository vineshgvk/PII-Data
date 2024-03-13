'''
data_download.py
downloads dataset from source and stashes the dataset in google cloud.
'''
import os
import json
import pandas as pd
from google.cloud import storage
import logging
import pickle

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
print(PROJECT_DIR)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(PROJECT_DIR, "key.json")

def load_data_from_gcp_and_save_as_pkl(pickle_path):
    try:
        # Define the destination directory for saving JSON files
        destination_dir = os.path.join(PROJECT_DIR, "data", "processed")

        # Copy specific files from GCP to the local directory
        files_to_copy = [
            "Data/train.json"
        ]

        # Create a client
        client = storage.Client()

        # Get the bucket
        bucket = client.get_bucket('pii_data_load')

        # List to store dictionaries (JSON contents)
        json_contents_list = []

        for file in files_to_copy:
            # Get the blob
            blob = storage.Blob(file, bucket)
            # Download the file to a destination
            blob.download_to_filename(os.path.join(destination_dir, os.path.basename(file)))
            # Read JSON file
            with open(os.path.join(destination_dir, os.path.basename(file)), 'r') as json_file:
                json_contents = json.load(json_file)
                json_contents_list.append(json_contents)

        # Save the list of JSON contents as a pickle file
        with open(pickle_path, 'wb') as f:
            pickle.dump(json_contents_list, f)

        print("Data loaded and saved as pkl successfully.")
    except Exception as e:
        logging.critical('data_download.py failed!')
        print(f"An unexpected error occurred: {e}")

# Specify the path where you want to save the pickle file
pickle_path = os.path.join(PROJECT_DIR, "data", "processed", "data.pkl")
print('pickle_path',pickle_path)
json_path = os.path.join(PROJECT_DIR, "data", "processed", "train.json")
print('json_path', json_path)

# def upload_blob(bucket_name, source_file_path, destination_blob_name):
#   """Uploads a file to the bucket."""
#   storage_client = storage.Client()

#   bucket = storage_client.get_bucket(bucket_name)
#   blob = bucket.blob(destination_blob_name)
#   #jsonSource = os.path.join(PROJECT_DIR, "data", "processed", "train.json")
#   blob.upload_from_filename(source_file_path)
#   print(f'Upload to Google cloud successful at {destination_blob_name}!')

# Call the function to load data from GCP and save it as pkl
load_data_from_gcp_and_save_as_pkl(pickle_path)
# pload_blob(bucket_name = 'processed_datafiles', source_file_path = json_path, destination_blob_name = 'files/md5/train.json')