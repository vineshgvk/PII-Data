
"""
A module for tokenizing the data and preparing the data for the model input format.
"""
import os
import pickle
import numpy as np
import logging
import pandas as pd
import json
import nltk
from nltk import word_tokenize

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
# PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.getcwd()

INPUT_LABELS_PKL_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','dup_removed.pkl')
OUTPUT_TOKEN_MAPPING_PKL_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','tokenized_data.pkl')

MODEL_PATH = 'microsoft/deberta-v3-base'
MAX_INFERENCE_LENGTH = 2048  # Specify the max length

def tokenizer(inputPath = INPUT_LABELS_PKL_PATH, 
              outputPath = OUTPUT_TOKEN_MAPPING_PKL_PATH):
    '''
    tokenizer
    Tokenize full_text column of the document through a nltk tokenizer (word and sentence) and stores it in nltkWordTokens column.
    Args:
        inputPath = input pickle location
        outputPath = output pickle location
    Returns:
        outputPath
    '''
    # Open file in read-binary mode if exists
    if os.path.exists(inputPath):
        with open(inputPath, "rb") as file:
            df = pd.read_pickle(file)
    else:
        logging.critical('FAILURE! tokenize_data.py failed')
        raise FileNotFoundError(f"FAILED! No such path at {inputPath}")

    # Tokenize 'full_text' and store it in nltkWordTokens column
    df['nltkWordTokens'] = df['full_text'].apply(word_tokenize)

    # Pickle dataset and push forward
    # opening file in write-binary mode
    with open(outputPath, "wb") as file:
        pickle.dump(df, file)
    print(f'Data pickled after tokenize_data at {outputPath}')

    return outputPath

tokenizer(inputPath = INPUT_LABELS_PKL_PATH, outputPath = OUTPUT_TOKEN_MAPPING_PKL_PATH)

# def tokenize_data(input_data_pkl=INPUT_DATA_PKL_PATH, input_labels_pkl=INPUT_LABELS_PKL_PATH, model_path=MODEL_PATH, output_tokenmapping_pkl=OUTPUT_TOKEN_MAPPING_PKL_PATH, max_inference_length=MAX_INFERENCE_LENGTH):
#     """
#     Tokenize data loaded from a pickle file using a pre-trained tokenizer, incorporating labels,
#     and then save the tokenized dataset to an output pickle file.

#     Args:
#         input_data_pkl (str): Path to the input pickle file containing the data.
#         input_labels_pkl (str): Path to the input pickle file containing the label mappings.
#         model_path (str): Path to the pre-trained model.
#         output_tokenmapping_pkl (str): Path to save the tokenized dataset.
#         max_inference_length (int): Maximum length of the tokenized sequences.

#     Raises:
#         FileNotFoundError: If the input data or labels pickle file is not found.
#     """
#     # Load data from the input pickle file
#     if not os.path.exists(input_data_pkl):
#         raise FileNotFoundError(f"Data file not found: {input_data_pkl}")
#     try:
#         with open(input_data_pkl, "rb") as file:
#             data = pickle.load(file)
#     except Exception as e:
#         logging.error('FAILURE! tokenize_data.py failed to load input pickle.')
#         raise Exception(f"Failed to load data from {input_data_pkl}: {e}")

#     # Load label mappings from the input labels pickle file
#     if not os.path.exists(input_labels_pkl):
#         logging.error('FAILURE! tokenize_data.py failed to load labels.')
#         raise FileNotFoundError(f"Labels file not found: {input_labels_pkl}")
#     try:
#         with open(input_labels_pkl, "rb") as file:
#             label2id = pickle.load(file)
#     except Exception as e:
#         raise Exception(f"Failed to load labels from {input_labels_pkl}: {e}")

#     # Define tokenization function that processes each example in the dataset
#     def tokenize(example, tokenizer, label2id, max_length):
#         text = []
#         labels = []

#         for t, l, ws in zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"]):
#             text.append(t)
#             labels.extend([l] * len(t))

#             if ws:
#                 text.append(" ")
#                 labels.append("O")

#         tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length)
#         labels = np.array(labels, dtype=object)

#         text = "".join(text)
#         token_labels = []

#         for start_idx, end_idx in tokenized.offset_mapping:
#             if start_idx == 0 and end_idx == 0:  # CLS token
#                 token_labels.append(label2id["O"])
#                 continue

#             if text[start_idx].isspace():  # Adjust for starting whitespace
#                 start_idx += 1

#             token_labels.append(label2id.get(labels[start_idx], label2id["O"]))  # Default to "O" if label not found

#         length = len(tokenized.input_ids)

#         return {**tokenized, "labels": token_labels, "length": length}

#     # Initialize the tokenizer from the specified pre-trained model
#     tokenizer = AutoTokenizer.from_pretrained(model_path)

#     # Apply the tokenization function to the dataset
#     ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": max_inference_length}, num_proc = 3)

#     # Save the tokenized dataset to the specified output pickle file
#     try:
#         ds.save_to_disk(output_tokenmapping_pkl)
#         print(f"Tokenized dataset saved to {output_tokenmapping_pkl} after tokenize_data.py")
#     except Exception as e:
#         logging.critical(f'FAILURE! Failed to save tokenized dataset at {output_tokenmapping_pkl}')
#         raise Exception(f"Failed to save the tokenized dataset to {output_tokenmapping_pkl}: {e}")
    
#     return output_tokenmapping_pkl

# tokenize_data(input_data_pkl=INPUT_DATA_PKL_PATH, input_labels_pkl=INPUT_LABELS_PKL_PATH, model_path=MODEL_PATH, output_tokenmapping_pkl=OUTPUT_TOKEN_MAPPING_PKL_PATH, max_inference_length=MAX_INFERENCE_LENGTH)