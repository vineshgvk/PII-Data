import os
import json
from unittest.mock import MagicMock, patch

import sys
# Append the path of the 'dags' directory to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.insert(0, parent_dir)

from resampling import resample_data

# Adjust paths according to your project structure
PROJECT_DIR = os.getcwd()
INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'duplicate_removal.pkl')
OUTPUT_JSON_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'resampled.json')



def test_resample_data_input_output(mocker):
    """
    Test checks if the input is a .pkl file and the output is a .json file.
    """
    # Mock the ti object and its xcom_pull method
    mocked_ti = MagicMock()
    mocked_ti.xcom_pull.return_value = INPUT_PICKLE_PATH

    # Mock json.dump to avoid actual file writing
    mocker.patch('json.dump')
    
    # Mock pandas.read_pickle instead of open, if that's what your function uses
    mocker.patch('pandas.read_pickle', return_value="Mocked DataFrame")

    # Execute the resample_data function
    result = resample_data(ti=mocked_ti)

    # Verify the output path
    assert result == OUTPUT_JSON_PATH, f"Expected {OUTPUT_JSON_PATH}, got {result}"

    # Verify json.dump was called, indicating writing to a JSON file
    assert json.dump.called, "json.dump was not called"

