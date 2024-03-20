import os
import pytest
import sys

from unittest.mock import MagicMock

# Append the path of the 'dags' directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.insert(0, parent_dir)

from dags.src.missing_values import naHandler  # Adjust the import path as needed

PROJECT_DIR = os.getcwd()
INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'train.json')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'missing_values.pkl')

# import os
# import pytest
# from unittest.mock import MagicMock
# from missing_values import naHandler

# # PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PROJECT_DIR = os.getcwd()
# INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'train_com.json')
# # OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR,  'processed', 'missing_values.pkl')

# OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'missing_values.pkl')


def test_naHandler_success(mocker):
    """
    Test successful removal of rows with missing values and saving of the dataframe.
    """
    # Mock the ti object and its xcom_pull method
    mocked_ti = MagicMock()
    mocked_ti.xcom_pull.return_value = INPUT_PICKLE_PATH

    # Use mocker to simulate passing the ti object
    result = naHandler(ti=mocked_ti)
    assert result == OUTPUT_PICKLE_PATH, f"Expected {OUTPUT_PICKLE_PATH}, but got {result}."

def test_naHandler_file_not_found(mocker):
    """
    Test that naHandler raises an error when the input pickle doesn't exist.
    """
    # Rename the input pickle temporarily to simulate its absence
    if os.path.exists(INPUT_PICKLE_PATH):
        os.rename(INPUT_PICKLE_PATH, INPUT_PICKLE_PATH + ".bak")

    # Mock the ti object and its xcom_pull method
    mocked_ti = MagicMock()
    mocked_ti.xcom_pull.return_value = INPUT_PICKLE_PATH

    with pytest.raises(FileNotFoundError):
        naHandler(ti=mocked_ti)

    # Rename the input pickle back to its original name
    if os.path.exists(INPUT_PICKLE_PATH + ".bak"):
        os.rename(INPUT_PICKLE_PATH + ".bak", INPUT_PICKLE_PATH)

