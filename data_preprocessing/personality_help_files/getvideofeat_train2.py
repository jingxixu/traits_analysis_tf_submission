#------------------------------------------------------------

import subprocess
import logging
import zipfile
import sys, os

sys.path.append('./help_scripts')
sys.path.append('./data')

import utils
import preprocessing_audiofeats as audioprocess
import preprocessing_videofeats_train2 as videoprocess
import progressbar

#------------------------------------------------------------

trainDataDownloader = './help_scripts/train_val_getDataDirect.py'
testDataDownloader = './help_scripts/test_getDataDirect.py'
audioPreprocessor = './data/preprocessing_audiofeats.py'
videoreprocessor = './data/preprocessing_videofeats.py'

#----------------------------------
# -- prepare the logger
FORMAT = '%(asctime)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
fileHandler = logging.FileHandler("{0}/{1}.log".format('.', 'setup'))
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(0)
logger.setLevel(0)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
consoleHandler.setLevel(0)
logger.addHandler(consoleHandler)
logger.info('downloading the training and validation data')

# download the training, validation, test data
#subprocess.call(['python', trainDataDownloader])
#subprocess.call(['python', testDataDownloader])

# extract the zip files of train, validation, test to appropriate directories
folders = ['train', 'validation'] #, 'test']

folder = 'train'
destfolder = os.path.join("/dvmm-filer2/datasets/PersonalityVideos/ECCV", folder)               
videoprocess.videoPreprocess(destfolder)

#subprocess.call(['python', audioPreprocessor])
#subprocess.call(['python', videoreprocessor])
