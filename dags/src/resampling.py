import json
import os
import pandas as pd
import pickle  # Import the pickle module

def resample_data(**kwargs):
    PROJECT_DIR = os.getcwd()
    print("fetched project directory successfully", PROJECT_DIR)    
    inputPath = os.path.join(PROJECT_DIR,"dags","processed","train.json")
    print("fetched the input file", inputPath) 
    try:
        # Load data from a JSON file
        with open(inputPath, "r") as json_file:
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
        output_file_path = os.path.join(PROJECT_DIR, "dags", "processed", "resampled.json")

        # Convert the processed DataFrame to a list of dicts (if not already in this format) and save as JSON
        with open(output_file_path, "w") as output_file:
            json.dump(data_n, output_file)

        print("Processed data saved successfully.", output_file_path)
        return output_file_path

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
