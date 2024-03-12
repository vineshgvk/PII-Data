
import pandas as pd
import os
import pickle
from google.cloud import storage

# Get the current working directory
PROJECT_DIR = os.getcwd()
print(PROJECT_DIR)
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'D:/OneDrive - Northeastern University/NEU/IE 7374 MLOPS/Final Project/PII_DATA_DETECTION/PII-Data/key.json'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(PROJECT_DIR, "key.json")
# D:\OneDrive - Northeastern University\NEU\IE 7374 MLOPS\Final Project\PII_DATA_DETECTION\PII-Data\key.json

def load_data_from_gcp_and_save_as_pkl(pickle_path):
    try:
        # Define the destination directory for saving CSV files
        destination_dir = os.path.join(PROJECT_DIR, "data", "processed")

        # Copy specific files from GCP to the local directory
        files_to_copy = [
            "Data_Jan/train_1.csv"

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
        combined_df = pd.concat(data_frames, ignore_index=True)

        # Save the combined DataFrame as a pickle file
        combined_df.to_pickle(pickle_path)

        print("Data loaded and saved as pkl successfully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Specify the path where you want to save the pickle file
pickle_path = os.path.join(PROJECT_DIR, "data", "processed", "data.pkl")
print(pickle_path)
# Call the function to load data from GCP and save it as pkl
load_data_from_gcp_and_save_as_pkl(pickle_path)
