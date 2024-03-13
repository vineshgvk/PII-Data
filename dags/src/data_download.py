import os
import json
from google.cloud import storage
import logging
# Get the current working directory
PROJECT_DIR = os.getcwd()
logging.info(PROJECT_DIR)

# Set the path to your Google Cloud service account key file
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(PROJECT_DIR, "key.json")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/lahariboni/Desktop/MLOps/Softwares/airflow_workspace/airflow/key.json'


# json_path=os.path.join(PROJECT_DIR, "data", "processed", "train.json")

def load_data_from_gcp_and_save_as_json():
    try:
        # Define the destination directory for saving JSON files
        destination_dir = os.path.join(PROJECT_DIR, "data", "processed")

        # Ensure the destination directory exists
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # Specify the files to copy from GCP to the local directory
        files_to_copy = [
            "Data/train.json"
        ]

        # Create a client for interacting with Google Cloud Storage
        client = storage.Client()

        # Specify your Google Cloud Storage bucket name
        bucket = client.get_bucket('pii_data_load')

        # Process and save each specified file
        for file in files_to_copy:
            # Define the blob (file in GCS)
            blob = storage.Blob(file, bucket)
            
            # Define the local filename path where the JSON will be saved
            local_file_path = os.path.join(destination_dir, os.path.basename(file))
            logging.info('works', local_file_path)
            # Download the file
            blob.download_to_filename(local_file_path)
            logging.error(local_file_path)
            print(f"File {os.path.basename(file)} downloaded and saved as {local_file_path}.")
        # return local_file_path
        return True

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Define the path where the JSON file will be saved (this could be adjusted as needed)
# json_path = os.path.join(PROJECT_DIR, "data", "processed", "train.json")

# Call the function to load data from GCP and save it as JSON
# load_data_from_gcp_and_save_as_json(json_path)



# import os
# import json
# import pandas as pd
# from google.cloud import storage
# import pickle

# # Get the current working directory
# PROJECT_DIR = os.getcwd()
# # print(PROJECT_DIR)

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(PROJECT_DIR, "key.json")

# def load_data_from_gcp_and_save_as_json(pickle_path):
#     try:
#         # Define the destination directory for saving JSON files
#         destination_dir = os.path.join(PROJECT_DIR, "data", "processed")

#         # Copy specific files from GCP to the local directory
#         files_to_copy = [
#             "Data/train.json"
#         ]

#         # Create a client
#         client = storage.Client()

#         # Get the bucket
#         bucket = client.get_bucket('pii_data_load')

#         # List to store dictionaries (JSON contents)
#         json_contents_list = []

#         for file in files_to_copy:
#             # Get the blob
#             blob = storage.Blob(file, bucket)
#             # Download the file to a destination
#             blob.download_to_filename(os.path.join(destination_dir, os.path.basename(file)))
#             # Read JSON file
#             with open(os.path.join(destination_dir, os.path.basename(file)), 'r') as json_file:
#                 json_contents = json.load(json_file)
#                 json_contents_list.append(json_contents)

#         # Save the list of JSON contents as a pickle file
#         with open(pickle_path, 'wb') as f:
#             pickle.dump(json_contents_list, f)

#         print("Data loaded and saved as pkl successfully.")
#         return pickle_path
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# # Specify the path where you want to save the pickle file
# pickle_path = os.path.join(PROJECT_DIR, "data", "processed", "train.json")
# # print(pickle_path)
# # Call the function to load data from GCP and save it as pkl
# # load_data_from_gcp_and_save_as_pkl(pickle_path)