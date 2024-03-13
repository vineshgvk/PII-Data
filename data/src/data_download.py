'''
data_download.py to download, load, and pickle the source dataset.
'''

import pandas as pd
import os
import pickle
from google.cloud import storage
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
print(f'Project Directory at {PROJECT_DIR}')
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'D:/OneDrive - Northeastern University/NEU/IE 7374 MLOPS/Final Project/PII_DATA_DETECTION/PII-Data/key.json'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(PROJECT_DIR, "key.json")
# D:\OneDrive - Northeastern University\NEU\IE 7374 MLOPS\Final Project\PII_DATA_DETECTION\PII-Data\key.json

def load_data_from_gcp_and_save_as_pkl(pickle_path):
    '''
    load_data_from_gcp_and_save_as_pkl
    Loads data from Google Cloud and pickles
    Args:
        pickle_path: Where to pickle the dataset
    Returns:
        None
    '''
    try:
        # Define the destination directory for saving CSV files
        destination_dir = os.path.join(PROJECT_DIR, "data", "processed")

        # Copy specific files from GCP to the local directory
        files_to_copy = [
            "Data_Jan/train_1.csv",
            # "Data_Feb/train_2.csv",
            # "Data_Mar/train_3.csv",
            # "Data_Apr/train_4.csv",
            # "Data_May/train_5.csv"
        ]

        # Create a client
        client = storage.Client()

        # Get the bucket
        bucket = client.get_bucket('pii_data_load')

        # List to store DataFrames
        data_frames = []

        for file in files_to_copy:
            # Get the blob
            blob = storage.Blob(file, bucket)
            # Download the file to a destination
            blob.download_to_filename(os.path.join(destination_dir, os.path.basename(file)))
            # Read CSV file into a DataFrame
            df = pd.read_csv(os.path.join(destination_dir, os.path.basename(file)))
            data_frames.append(df)

        # Concatenate all DataFrames into one
        combined_df = pd.concat(data_frames, ignore_index = True)

        # Save the combined DataFrame as a pickle file
        combined_df.to_pickle(pickle_path)

        print(f'SUCCESS! Data pickled at {pickle_path}')
    except Exception as e:
        logging.critical('data_download.py failed!')
        print(f"FAILURE! An unexpected error occurred: {e}")

# Specify the path where you want to save the pickle file
pickle_path = os.path.join(PROJECT_DIR, "data", "processed", "data.pkl")
print(f'Pickle path at {pickle_path}')

def upload_to_bucket(file_path = pickle_path, bucket_name = 'processed_datafiles'):
    storage_client = storage.Client()
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob('data-processed.pkl')
        blob.upload_from_filename(file_path)
        print(bucket)
        return True
    except Exception as e:
        logging.error('FAILURE! Could not upload to GCS bucket')
        print('Upload to GCS bucket failed!')
        return False

# Call the function to load data from GCP and save it as pkl
if __name__ == "__main__":
    load_data_from_gcp_and_save_as_pkl(pickle_path)
    upload_to_bucket(file_path = pickle_path, bucket_name = 'processed_datafiles')