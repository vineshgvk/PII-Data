"""
A module for label encoding the target column.
"""

import os
import pickle
from itertools import chain



# Determine the absolute path of the project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DATA_PKL_PATH=os.path.join(PROJECT_DIR, 'data', 'processed','duplicate_removed.pkl')
OUTPUT_LABELS_PKL_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','label_encoder_data.pkl')
               

def load_and_transform_labels_from_pkl(input_pkl_path=INPUT_DATA_PKL_PATH, output_pickle_path=OUTPUT_LABELS_PKL_PATH):
    """
    Load data from the input pickle file, compute label2id and id2label mappings,
    and save them to the output pickle file.

    :param input_pkl_path: Path to the input pickle file.
    :param output_pickle_path: Path to the output pickle file.
    """
    # Check if the input file exists
    if not os.path.exists(input_pkl_path):
        raise FileNotFoundError(f"No data found at the specified path: {input_pkl_path}")
    
    # Load data from input pickle file
    with open(input_pkl_path, "rb") as file:
        data = pickle.load(file)
    
    # Compute all_labels, label2id, and id2label
    all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}
    
    # Data to be saved
    label_encoder_data = {
        "label2id": label2id,
        "id2label": id2label
    }
    
    # Save the mappings to the output pickle file
    with open(output_pickle_path, "wb") as file:
        pickle.dump(label_encoder_data, file)
    
    print(f"Data saved to {output_pickle_path}.")
    
    return output_pickle_path


