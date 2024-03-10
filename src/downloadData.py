'''
downloadData
downloads zipped datasets from source.
gdown docs: https://github.com/wkentaro/gdown
'''
import os
import gdown
import logging

# Stash the logs in the src/logs path.
logging.basicConfig(filename = 'logs/logs.log', # log filename with today's date.
                    filemode = "w", # write mode
                    level = logging.ERROR, # Set error as the default log level.
                    format ='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # logging format
                    datefmt = '%Y-%m-%d %H:%M:%S',) # logging (asctime) date format

# Data stored in GDrive.
BASE_PATH = 'https://drive.google.com/uc?id=13S0QO_0K6lfoIIGTLA_E6z45WuGyFT7s'

def downloadData(downloadURL = BASE_PATH):
    '''
    downloadData: To download source datasets from Google Drive.
    Args:
        downloadURL: URL with Google Drive ID link.
    Returns:
        zipPath: Zipped file path of the downloaded dataset.
    '''
    rootDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    zipPath = os.path.join(rootDir, 'data','pii-data-zip.zip')
    
    # Remove os path if exists.
    if os.path.exists(zipPath):
        os.remove(zipPath)

    gdown.download(url = downloadURL, 
                   output = zipPath, 
                   quiet = False)
    
    if os.path.exists(zipPath):
        print(f'SUCCESS! File downloaded to {zipPath}')
    else:
        print('FAILED! Check again')
        logging.critical('downloadData.py failed!')
    
    return zipPath

if __name__ == "__main__":
    zipPath = downloadData(downloadURL = 'https://drive.google.com/uc?id=13S0QO_0K6lfoIIGTLA_E6z45WuGyFT7s')