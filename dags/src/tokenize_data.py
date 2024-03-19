import os
import json
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
import pickle


def tokenize_data(**kwargs):
    '''
    tokenize_data
    '''
    PROJECT_DIR = os.getcwd()
    print("fetched project directory successfully",PROJECT_DIR)    
    ti_r = kwargs['ti']
    data_path = ti_r.xcom_pull(task_ids='resample_data')
    print("fetched path from resample_data task",data_path)
    
    ti_l = kwargs['ti']
    label2id_path = ti_l.xcom_pull(task_ids='label_encoder')
    print("fetched path from resample_data task",label2id_path)

    # Load label2id from JSON file

    output_tokenized = os.path.join(PROJECT_DIR, 'dags', 'processed', 'tokenized_data.pkl')
    model_path = 'microsoft/deberta-v3-base'
    max_inference_length = 2048
    with open(label2id_path, "r") as file:
        label_encoder_data = json.load(file)
        label2id = label_encoder_data["label2id"]
        print("labelencoder data has been fectched successfully")

    # Load data from JSON file
    with open(data_path, "r") as file:
        data = json.load(file)
        print("resampled data has been fectched successfully")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Inititated the tokenizer using the model", model_path)

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
    ds = ds.map(tokenize, num_proc=1)

    # Save tokenized dataset to JSON file
    with open(output_tokenized,"wb") as f:
        pickle.dump(ds,f)

    return output_tokenized

