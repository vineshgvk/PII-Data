import os
import pytest
import sys
from unittest.mock import MagicMock
# Append the path of the 'dags' directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.insert(0, parent_dir)


from duplicates import dupeRemoval

PROJECT_DIR = os.getcwd()
INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'missing_values.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'duplicate_removal.pkl')


def test_dupeRemoval_file_found(mocker):
    """
    Test that dupeRemoval raises an error when the input pickle doesn't exist.
    """
    # Rename the input pickle temporarily to simulate its absence
    if os.path.exists(INPUT_PICKLE_PATH):
        os.rename(INPUT_PICKLE_PATH, INPUT_PICKLE_PATH + ".bak")

    # Mock the ti object and its xcom_pull method
    mocked_ti = MagicMock()
    mocked_ti.xcom_pull.return_value = INPUT_PICKLE_PATH

    with pytest.raises(FileNotFoundError):
        dupeRemoval(ti=mocked_ti)

    # Rename the input pickle back to its original name
    if os.path.exists(INPUT_PICKLE_PATH + ".bak"):
        os.rename(INPUT_PICKLE_PATH + ".bak", INPUT_PICKLE_PATH)

def test_dupeRemoval_success(mocker):
    """
    Test successful removal of duplicate rows and saving of the dataframe.
    """
    # Mock the ti object and its xcom_pull method
    mocked_ti = MagicMock()
    mocked_ti.xcom_pull.return_value = INPUT_PICKLE_PATH
    print("Called")
    # Use mocker to simulate passing the ti object
    result = dupeRemoval(ti=mocked_ti)
    assert result == OUTPUT_PICKLE_PATH, f"Expected {OUTPUT_PICKLE_PATH}, but got {result}."

