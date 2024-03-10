'''
dupeRemoval
Removes duplicate, if any, from the dataset.
loadData > naHandler (returns afterNA.pkl) > dupeRemoval (returns afterDupeRemoval.pkl)
'''
import os
import pickle

rootDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# pklPath is input to function. outPklPath is path after processing.
pklPath = os.path.join(rootDir, 'data', 
                       'processed', 'afterNA.pkl')
outPklPath = os.path.join(rootDir, 'data', 
                          'processed','afterDupeRemoval.pkl')

def dupeRemoval(inputPath = pklPath, 
                outputPath = outPklPath):
    '''
    dupeRemoval checks for duplicates in the dataset and removes any.
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
    
    # Removes rows i.e. any duplicates in the full_text column
    df.drop_duplicates(subset = ['full_text'],
                       inplace = True)
    
    # Pickle dataset and push forward
    # opening file in write-binary mode
    with open(outputPath, "wb") as file:
        pickle.dump(df, file)
    print(f'Data pickled after dupeRemoval at {outputPath}')

    return outputPath

dupeRemoval(inputPath = pklPath, outputPath = outPklPath)