'''
missing_values.py
Looks for N/As in the dataset and removes relevent records accordingly.
data_download > missing_values (returns missing_values.pkl)
'''
import pickle
import os

# Get the current working directory
PROJECT_DIR = os.getcwd()
# pklPath is input to function. outPklPath is path after processing.
pklPath = os.path.join(PROJECT_DIR, 'data.pkl')
outPklPath = os.path.join(PROJECT_DIR, 'missing_values.pkl')

def naHandler(inputPath=pklPath, outputPath=outPklPath):
    if os.path.exists(inputPath):
        print("Loading data from:", inputPath)
        with open(inputPath, "rb") as file:
            df = pickle.load(file)
    else:
        raise FileNotFoundError(f"FAILED! No such path at {inputPath}")

    print("Original DataFrame:")
    print(df.head())

    # Remove NAs wherever applicable.
    df.dropna(subset=['full_text'], inplace=True)

    print("DataFrame after removing missing values:")
    print(df.head())

    nullCount = df.isnull().sum().sum()
    if nullCount > 0:
        nullsPresentError = f'Nulls {nullCount} still present in the dataset'
        print(nullsPresentError)
        raise ValueError(nullsPresentError)

    # Pickle dataset and push forward
    with open(outputPath, "wb") as file:
        pickle.dump(df, file)
    print(f'Data pickled after naHandling at {outputPath}')

    return outputPath

naHandler(inputPath=pklPath, outputPath=outPklPath)