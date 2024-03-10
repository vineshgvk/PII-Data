'''
anomalyDetect
Checks for the right data types, looks for text length below threshold.
loadData > naHandler > dupeRemoval (returns afterDupeRemoval.pkl) > anomalyDetect (returns afterAnomalyDetection.pkl)
'''
import pickle
import os
import logging

rootDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# pklPath is input to function. outPklPath is path after processing.
pklPath = os.path.join(rootDir, 'data', 
                       'processed', 'afterDupeRemoval.pkl')
outPklPath = os.path.join(rootDir, 'data', 
                          'processed','afterAnomalyDetection.pkl')

textThreshold = 25 # Remove records with full_text < 25 words.
trainSamples = 100 # Needs at least trainSamples amount of records for training.
expectedDtypes = {'document': int,
                  'full_text': object,
                  'tokens': object,
                  'trailing_whitespace': object,
                  'labels': object
                  }

def anomalyDetect(inputPath = pklPath, 
                outputPath = outPklPath,
                textThreshold = textThreshold,
                trainSamples = trainSamples,
                expectedDtypes = expectedDtypes):
    '''
    anomalyDetect looks for the right data types, and checks for text length below a threshold
    Args:
        inputPath: Input pickle path after loadData.
        outputPath: Output pickle path after dupeRemoval processing.
    Returns:
        outputPath
    '''

    # Open file in read-binary mode if exists
    if os.path.exists(inputPath):
        with open(inputPath, "rb") as file:
            df = pickle.load(file)
    else:
        raise FileNotFoundError(f"FAILED! No such path at {inputPath}")

    # Check for text length
    rowsRemoved = 0
    for index, row in df.iterrows():
        if len(row['full_text'].split()) < textThreshold:
            rowsRemoved += 1
            df.drop(index, inplace = True)
    print(f'Records removed because of text length threshold {textThreshold}: {rowsRemoved} records')

    # Check for trainSamples threshold for training
    if df.shape[0] < trainSamples:
        print(f'Not enough training samples for model to be trained')

    # Check for appropriate text types
    for col in df.columns:
        if df[col].dtype != expectedDtypes[col]:
            print(f'{col} data type mismatch')

    # Pickle dataset and push forward
    # opening file in write-binary mode
    with open(outputPath, "wb") as file:
        pickle.dump(df, file)
    print(f'Data pickled after anomalyDetect at {outputPath}')

    return outputPath

# anomalyDetect(inputPath = pklPath, outputPath = outPklPath, textThreshold = 25, expectedDtypes = expectedDtypes)