'''
unzipFile
Unzips downloaded zip file.
'''
import os
import zipfile
import logging
from datetime import datetime

logsPath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs'))
if not os.path.exists(logsPath):
    # Create the folder if it doesn't exist
    os.makedirs(logsPath)
    print(f"Folder '{logsPath}' created successfully.")

# Stash the logs in the src/logs path.
logging.basicConfig(filename = 'logs/logs.log', # log filename with today's date.
                    filemode = "w", # write mode
                    level = logging.ERROR, # Set error as the default log level.
                    format ='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # logging format
                    datefmt = '%Y-%m-%d %H:%M:%S',) # logging (asctime) date format

# zipPath should contain zipped dataset; extractPath to contain the unzipped files
rootDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
zipPath = os.path.join(rootDir, 'data','pii-data-zip.zip')
extractPath = os.path.join(rootDir,'data')

def unzipFile(zipPath = zipPath, extractPath = extractPath):
    """
    Function to unzip the downloaded data
    Args:
      zipPath: zipfile path, a default is used if not specified
      extractPath: Path where the unzipped and extracted data is available
    Returns:
      extractPath: filepath where the data is available
    """
    try:
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall(extractPath)
        print(f"File {zipPath} successfully unzipped to {extractPath}")
    except zipfile.BadZipFile:
        logging.critical('unzipFile.py failed!')
        print(f"Failed to unzip {zipPath}")
    # Return unzipped file
    return os.path.join(extractPath)

if __name__ == "__main__":
    unzipPath = unzipFile(zipPath = zipPath, extractPath = extractPath)