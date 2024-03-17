import json
import os
import pandas as pd
import pickle  # Import the pickle module

def resample_data(**kwargs):
    PROJECT_DIR = os.getcwd()
    print("fetched project directory successfully", PROJECT_DIR)    
    ti = kwargs['ti']
    inputPath = ti.xcom_pull(task_ids='load_data_from_gcp')
    print("fetched path from load_gcp_data task", inputPath) 
    try:
        # Load data from a pickle file instead of a JSON file
        with open(inputPath, "rb") as pickle_file:  # Note the "rb" mode for binary files
            data = pickle.load(pickle_file)

        # Assuming 'data' is a DataFrame and contains a 'labels' column.
        # Adjust the logic here depending on the actual structure of your data.
        
        # Downsampling of negative examples
        p = []  # positive samples (contain relevant labels)
        n = []  # negative samples (presumably contain entities that are possibly wrongly classified as entity)
        for index, row in data.iterrows():
            if any(label != "O" for label in row['labels']):
                p.append(row)
            else:
                n.append(row)
        
        # Combine data - assuming 'data' is a DataFrame. Convert lists to DataFrames if necessary.
        data_n = pd.concat([data, pd.DataFrame(p), pd.DataFrame(n[:len(n) // 3])])

        # Specify the output file path
        output_file_path = os.path.join(PROJECT_DIR, "dags", "processed", "resampled.json")

        # Convert the processed DataFrame to a list of dicts (if not already in this format) and save as JSON
        with open(output_file_path, "w") as output_file:
            json.dump(data_n.to_dict(orient='records'), output_file)

        print("Processed data saved successfully.", output_file_path)
        return output_file_path

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
