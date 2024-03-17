import os
import json
import pytest
from unittest.mock import MagicMock


import sys
# Append the path of the 'dags' directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.insert(0, parent_dir)
from label_encoder import target_label_encoder  # Adjust the import path as needed

PROJECT_DIR = os.getcwd()
INPUT_JSON_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'resampled.json')
OUTPUT_JSON_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'label_encoder_data.json')

def setup_module(module):
    """Setup for the entire module"""
    data = [
        {'labels': ['B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS',
                    'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM',
                    'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O']},
        # Add more entries if needed
    ]
    with open(INPUT_JSON_PATH, "w") as file:
        json.dump(data, file)

def teardown_module(module):
    """Teardown for the entire module"""
    os.remove(INPUT_JSON_PATH)
    if os.path.exists(OUTPUT_JSON_PATH):
        os.remove(OUTPUT_JSON_PATH)

def test_target_label_encoder_success():
    """
    Test that target_label_encoder successfully processes an existing input file and creates an output file.
    """
    mocked_ti = {'ti': MagicMock()}
    mocked_ti['ti'].xcom_pull.return_value = INPUT_JSON_PATH

    result = target_label_encoder(**mocked_ti)

    assert result == OUTPUT_JSON_PATH, f"Expected {OUTPUT_JSON_PATH}, but got {result}."
    assert os.path.exists(OUTPUT_JSON_PATH), "Output file was not created."

    with open(OUTPUT_JSON_PATH, "r") as file:
        data = json.load(file)
        assert 'label2id' in data and 'id2label' in data, "Output file does not contain expected mappings."

def test_target_label_encoder_unexpected_labels():
    """
    Test that target_label_encoder handles unexpected labels properly.
    """
    # Add an entry with an unexpected label
    unexpected_data = [
        {'labels': ['UNEXPECTED_LABEL']}
    ]
    with open(INPUT_JSON_PATH, "w") as file:
        json.dump(unexpected_data, file)

    mocked_ti = {'ti': MagicMock()}
    mocked_ti['ti'].xcom_pull.return_value = INPUT_JSON_PATH

    # Here you might expect an exception or handle it depending on your implementation
    # For illustration, let's assume we just run the function without expecting an error
    # Adjust this part based on your actual error handling strategy
    result = target_label_encoder(**mocked_ti)

    # Check if the function still outputs a file or not, depending on expected behavior
    # This example assumes the function continues without error and creates an output file
    assert os.path.exists(OUTPUT_JSON_PATH), "Output file was not created despite unexpected labels."
    # You might also check the content of the output file to see how it handled the unexpected label

# Note: The `test_target_label_encoder_unexpected_labels` test needs to be adjusted based on how your function is supposed to handle unexpected labels. If your function should raise an exception for unexpected labels, you would use `with pytest.raises(SomeException):` to wrap the function call.



# import os
# import json
# import pytest
# from unittest.mock import MagicMock, mock_open, patch
# from label_encoder import target_label_encoder  # Adjust the import path as needed

# PROJECT_DIR = os.getcwd()
# INPUT_JSON_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'resampled.json')
# OUTPUT_JSON_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'label_encoder_data.json')

# def test_target_label_encoder_success(mocker):
#     """
#     Test that target_label_encoder correctly processes an existing input file and creates an output file.
#     """
#     # Prepare a generic mock data string with a variable number of labels
#     mock_data = json.dumps([{"labels": ["label1", "label2", "label3", "..."]}])  # Extendable to any number of labels

#     # Mock the ti object and its xcom_pull method to return the input JSON path
#     mocked_ti = MagicMock()
#     mocked_ti.xcom_pull.return_value = INPUT_JSON_PATH

#     # Use mock_open with a side_effect to handle multiline JSON correctly
#     mocker.patch('builtins.open', mock_open(read_data=mock_data), create=True)

#     # Mock 'json.dump' to prevent actual file writing
#     mocker.patch('json.dump')

#     # Execute the function
#     result = target_label_encoder(ti=mocked_ti)

#     # Assert that the output path is as expected
#     assert result == OUTPUT_JSON_PATH, f"Expected {OUTPUT_JSON_PATH}, but got {result}."

#     # Verify that 'json.dump' was called, implying the output file was created
#     json.dump.assert_called_once()
