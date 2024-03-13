import subprocess
import pandas as pd
import os
import pickle

def load_data_from_gcp_and_save_as_pkl(pickle_path):
    try:
        # Define the destination directory for saving CSV files
        destination_dir = os.path.join(PROJECT_DIR, "gcp")

        # Copy specific files from GCP to the local directory
        files_to_copy = [
            "Data_Jan/train_1.csv",
            "Data_Feb/train_2.csv",
            "Data_Mar/train_3.csv",
            "Data_Apr/train_4.csv",
            "Data_May/train_5.csv"
        ]

        # List to store DataFrames
        data_frames = []

        for file in files_to_copy:
            subprocess.run(["gsutil", "cp", f"gs://pii_data_load/{file}", destination_dir], check=True, capture_output=True)
            # Read CSV file into a DataFrame
            local_file_path = os.path.join(destination_dir, os.path.basename(file))
            df = pd.read_csv(local_file_path)
            data_frames.append(df)

        # Concatenate all DataFrames into one
        combined_df = pd.concat(data_frames, ignore_index=True)

        # Save the combined DataFrame as a pickle file
        combined_df.to_pickle(pickle_path)

        print("Data loaded and saved as pkl successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing gsutil command: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except PermissionError as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Get the current working directory
PROJECT_DIR = os.getcwd()

# Specify the path where you want to save the pickle file
pickle_path = os.path.join(PROJECT_DIR, 'data.pkl')

# Call the function to load data from GCP and save it as pkl
load_data_from_gcp_and_save_as_pkl(pickle_path)