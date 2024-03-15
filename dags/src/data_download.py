import os
import json
from google.cloud import storage
import logging
import NotFound
# Get the current working directory
# PROJECT_DIR = os.getcwd()
# logging.info(PROJECT_DIR)

# Set the path to your Google Cloud service account key file
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(PROJECT_DIR, "key.json")
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/vineshgvk/PII-Data/key.json'


# json_path=os.path.join(PROJECT_DIR, "data", "processed", "train.json")

def load_data_from_gcp_and_save_as_json():
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/vineshgvk/PII-Data/key.json'
        print("environment path is set")
        PROJECT_DIR = os.getcwd()
        
        print("current project directory",PROJECT_DIR)
        # Define the destination directory for saving JSON files
        destination_dir = os.path.join(PROJECT_DIR,"dags", "processed")
        print("destination directory at",destination_dir)

        # Ensure the destination directory exists
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
            print("Creating a new Destination directory since there is no existence")

        # Specify the files to copy from GCP to the local directory
        files_to_copy = [
            "Data/train_com.json"
        ]
        print("Trying to create a storage client connection")
        # Create a client for interacting with Google Cloud Storage
        client = storage.Client()
        print("Trying to fetch the specified bucket name")
        # try:
        #     bucket = client.get_bucket(bucket_name)
        #     print("Specified bucket is successfully fetched")
            
        # except NotFound:
        #     print(f"The bucket {bucket_name} was not found.")
        # Specify your Google Cloud Storage bucket name
    #     bucket = client.get_bucket(bucket_name)
        
        bucket = client.get_bucket('pii_data_load')
        print("Specified bucket is successfully fetched")
        

        # Process and save each specified file
        for file in files_to_copy:
            print("Starting iteration on files from bucket")
            # Define the blob (file in GCS)
            blob = storage.Blob(file, bucket)
            print("defined the blob to fetch",blob)
            # Define the local filename path where the JSON will be saved
            local_file_path = os.path.join(destination_dir, os.path.basename(file))
            print("local file path is created at", local_file_path)
            logging.info('works', local_file_path)
            # Download the file
            blob.download_to_filename(local_file_path)
            # logging.error(local_file_path)
            print(f"File {os.path.basename(file)} downloaded and saved as {local_file_path}.")
        # return local_file_path
        return local_file_path

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
