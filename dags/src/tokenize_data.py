import os
import json
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset

# Determine the absolute path of the project directory
PROJECT_DIR = os.getcwd()

LABEL2ID_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'label_encoder_data.json')
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'resampled.json')
OUTPUT_TOKENIZED_JSON_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'tokenized_data.json')

# Assuming these variables are defined elsewhere
TRAINING_MODEL_PATH = 'microsoft/deberta-v3-base'
TRAINING_MAX_LENGTH = 2048

def tokenize_data(data_path=DATA_PATH, label2id_path=LABEL2ID_PATH, output_tokenized_json_path=OUTPUT_TOKENIZED_JSON_PATH, model_path=TRAINING_MODEL_PATH, max_inference_length=TRAINING_MAX_LENGTH):
    # Load label2id from JSON file
    with open(label2id_path, "r") as file:
        label_encoder_data = json.load(file)
        label2id = label_encoder_data["label2id"]

    # Load data from JSON file
    with open(data_path, "r") as file:
        data = json.load(file)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Define tokenization function
    def tokenize(example):
        text = []
        labels = []

        for t, l, ws in zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"]):
            text.append(t)
            labels.extend([l] * len(t))

            if ws:
                text.append(" ")
                labels.append("O")

        tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_inference_length)

        labels = np.array(labels)

        text = "".join(text)
        token_labels = []

        for start_idx, end_idx in tokenized.offset_mapping:
            if start_idx == 0 and end_idx == 0:
                token_labels.append(label2id["O"])
                continue

            if text[start_idx].isspace():
                start_idx += 1

            token_labels.append(label2id[labels[start_idx]])

        length = len(tokenized.input_ids)

        return {**tokenized, "labels": token_labels, "length": length}

    # Create Dataset from data
    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [str(x["document"]) for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        "provided_labels": [x["labels"] for x in data],
    })

    # Tokenize dataset
    ds = ds.map(tokenize, num_proc=3)

    # Save tokenized dataset to JSON file
    ds.to_json(output_tokenized_json_path)

    return ds

# Tokenize data and get tokenized dataset
# tokenized_ds = tokenize_data()

