"""
Load data from the input JSON file, compute label2id and id2label mappings,
and save them to the output JSON file.

:param input_json_path: Path to the input JSON file.
:param output_json_path: Path to the output JSON file.
"""
import os
import json
from itertools import chain
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

# Determine the absolute path of the project directory
PROJECT_DIR = os.getcwd()

INPUT_DATA_JSON_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'resampled.json')
OUTPUT_LABELS_JSON_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'label_encoder_data.json')


def load_and_transform_labels_from_json(input_json_path=INPUT_DATA_JSON_PATH, output_json_path=OUTPUT_LABELS_JSON_PATH):
    """
    Load data from the input JSON file, compute label2id and id2label mappings,
    and save them to the output JSON file.

    :param input_json_path: Path to the input JSON file.
    :param output_json_path: Path to the output JSON file.
    """
    # Check if the input file exists
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"No data found at the specified path: {input_json_path}")

    # Load data from input JSON file
    with open(input_json_path, "r") as file:
        data = json.load(file)

    if not isinstance(data, list):
        logging.critical('Failure! Loaded data is not in the expected list format.')
        raise ValueError("Loaded data is not in the expected list format.")

    # Assuming each item in the list contains a 'labels' key with a list of labels
    all_labels_list = [item["labels"] for item in data]
    all_labels = sorted(list(set(chain(*all_labels_list))))

    # Compute label2id and id2label mappings
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Data to be saved
    label_encoder_data = {
        "label2id": label2id,
        "id2label": id2label
    }

    # Save the mappings to the output JSON file
    with open(output_json_path, "w") as file:
        json.dump(label_encoder_data, file, indent=4)

    print(f"Data saved to {output_json_path}.")

    return output_json_path


load_and_transform_labels_from_json(input_json_path=INPUT_DATA_JSON_PATH, output_json_path=OUTPUT_LABELS_JSON_PATH)
