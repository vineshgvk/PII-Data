"""
Load data from the input JSON file, validate datatypes and formats.
:param input_json_path: Path to the input JSON file.
"""
import json
import pandas as pd
import logging
def anomalyDetect(**kwargs):
    ti = kwargs['ti']
    input_path,_,_ = ti.xcom_pull(task_ids='data_slicing_batches_task')  # Get the output of the prior task
    with open(input_path, "r") as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    results = {
        "stats": {},
        "issues": {}
    }
    # Calculate statistics
    results['stats']['average_text_length'] = df['full_text'].apply(lambda x: len(x.split())).mean()
    results['stats']['average_token_length'] = df['tokens'].apply(len).mean()
    results['stats']['num_rows'] = df.shape[0]
    results['stats']['num_columns'] = df.shape[1]
    # Expected norms and types
    expected_text_length = 500
    expected_token_length = 500
    min_rows = 10
    # Check discrepancies
    if results['stats']['average_text_length'] != expected_text_length:
        results['issues']['text_length_issue'] = f"Expected average text length {expected_text_length}, found {results['stats']['average_text_length']}"
        logging.error(f"Expected average text length {expected_text_length}, found {results['stats']['average_text_length']}")
    if results['stats']['average_token_length'] < expected_token_length:
        results['issues']['token_length_issue'] = f"Expected minimum average token length {expected_token_length}, found {results['stats']['average_token_length']}"
        logging.error(f"Expected minimum average token length {expected_token_length}, found {results['stats']['average_token_length']}")
    if results['stats']['num_rows'] < min_rows:
        results['issues']['row_count_issue'] = f"Insufficient data rows for analysis, expected at least {min_rows}, found {results['stats']['num_rows']}"
        logging.error(f"Insufficient data rows for analysis, expected at least {min_rows}, found {results['stats']['num_rows']}")
    # Data type and value checks
    expected_dtypes = {'document': 'int64', 'full_text': 'object', 'tokens': 'object',
                       'trailing_whitespace': 'object', 'labels': 'object'}
    allowed_labels = ['B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS',
                      'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM',
                      'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O']
    for column, expected_dtype in expected_dtypes.items():
        if column in df.columns:
            if df[column].dtype != expected_dtype:
                results['issues'][f'{column}_dtype_issue'] = f"Expected {expected_dtype}, found {df[column].dtype}"
                logging.error(f"Expected {expected_dtype}, found {df[column].dtype}")
        else:
            results['issues'][f'{column}_missing'] = "Column is missing in the dataset"
            logging.error(f"Column {column} is missing in the dataset")
    # if 'trailing_whitespace' in df.columns and not df['trailing_whitespace'].isin([True, False]).all():
    #     results['issues']['trailing_whitespace_values'] = "Contains values other than True or False"
    if 'trailing_whitespace' in df.columns:
        if not all(df['trailing_whitespace'].map(lambda x: all(isinstance(i, bool) for i in x))):
            results['issues']['trailing_whitespace_values'] = "One or more lists in 'trailing_whitespace' column contains non-boolean values."
            logging.error("One or more lists in trailing_whitespace' column contains non-boolean values.")
    else:
        results['issues']['trailing_whitespace_missing'] = "'trailing_whitespace' column is missing."
        logging.error("'trailing_whitespace' column is missing.")
    if 'labels' in df.columns:
        flat_labels = df['labels'].explode().unique()
        invalid_labels = [label for label in flat_labels if label not in allowed_labels]
        if invalid_labels:
            results['issues']['invalid_labels'] = f"Invalid labels detected: {invalid_labels}"
            logging.error(f"Invalid labels detected: {invalid_labels}")
    else:
        results['issues']['labels_missing'] = "The 'labels' column is missing"
        logging.error("The 'labels' column is missing")
    return results


