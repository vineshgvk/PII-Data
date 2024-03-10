'''
naHandler
Looks for N/As in the dataset and removes relevent records accordingly.
loadData > naHandler (returns afterNA.pkl)
'''
import pickle
import os

rootDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# pklPath is input to function. outPklPath is path after processing.
pklPath = os.path.join(rootDir, 'data', 
                       'processed', 'rawData.pkl')
outPklPath = os.path.join(rootDir, 'data', 
                          'processed','afterNA.pkl')

def naHandler(inputPath = pklPath,
              outputPath = outPklPath):
    '''
    naHandler checks for NULLs in the dataset and handles appropriately.
    Args:
        inputPath: Input pickle path after loadData.
        outputPath: Output pickle path after naHandler processing.
    Returns:
        outputPath
    '''
    
    # Open file in read-binary mode if exists
    if os.path.exists(inputPath):
        with open(inputPath, "rb") as file:
            df = pickle.load(file)
    else:
        raise FileNotFoundError(f"FAILED! No such path at {inputPath}")
    
    # Remove NAs wherever applicable.
    df.dropna(subset = ['full_text'],
              inplace = True)
    
    nullCount = df.isnull().sum().sum()
    if nullCount > 0:
        nullsPresentError= f'Nulls {nullCount} still present in the dataset'
        print(nullsPresentError)
        raise ValueError(nullsPresentError)
    
    # Pickle dataset and push forward
    # opening file in write-binary mode
    with open(outputPath, "wb") as file:
        pickle.dump(df, file)
    print(f'Data pickled after naHandling at {outputPath}')

    return outputPath

# naHandler(inputPath = pklPath, outputPath = outPklPath)