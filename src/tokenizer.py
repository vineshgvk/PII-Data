'''
tokenizer
Tokenize full_text column of the document through a nltk tokenizer (word and sentence) and stores it in nltkWordTokens column.
loadData > naHandler > dupeRemoval > anomalyDetect (returns afterAnomalyDetection.pkl) > tokenizer (returns afterTokenizer.pkl)
'''
import pickle
import os
from nltk import word_tokenize
import nltk
nltk.download('punkt')

rootDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# pklPath is input to function. outPklPath is path after processing.
pklPath = os.path.join(rootDir, 'data', 
                       'processed', 'afterAnomalyDetection.pkl')
outPklPath = os.path.join(rootDir, 'data', 
                          'processed','afterTokenizer.pkl')

def tokenizer(inputPath = pklPath, 
              outputPath = outPklPath):
    '''
    tokenizer
    Tokenize full_text column of the document through a nltk tokenizer (word and sentence) and stores it in nltkWordTokens column.
    Args:
        inputPath = input pickle location
        outputPath = output pickle location
    Returns:
        outputPath
    '''
    # Open file in read-binary mode if exists
    if os.path.exists(inputPath):
        with open(inputPath, "rb") as file:
            df = pickle.load(file)
    else:
        raise FileNotFoundError(f"FAILED! No such path at {inputPath}")
    
    # Tokenize 'full_text' and store it in nltkWordTokens column
    df['nltkWordTokens'] = df['full_text'].apply(word_tokenize)

    # Pickle dataset and push forward
    # opening file in write-binary mode
    with open(outputPath, "wb") as file:
        pickle.dump(df, file)
    print(f'Data pickled after anomalyDetect at {outputPath}')

    return outputPath

# tokenizer(inputPath = pklPath, outputPath = outPklPath)