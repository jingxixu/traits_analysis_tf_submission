'''

----------------------------------------------------------------------
-- Get landmark and AU features for ECCV dataset 
-- to /data/junting/ECCV/trainlandmarkAU 
-- and /data/junting/ECCV/validationlandmarkAU

-- Jingxi Xu
----------------------------------------------------------------------

'''



import pandas as pd
import numpy as np
import os
import subprocess

# Make the graphs a bit prettier, and bigger
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60)


### Some paths
OpenfaceBase = "/home/jingxi/first-impressions/data/OpenFace/build/bin"
sourcefolder = "/dvmm-filer2/datasets/PersonalityVideos/ECCV"
destfolder = "/dvmm-filer2/datasets/PersonalityVideos/ECCV"

### all frame folders
validation_framefolders = os.listdir(os.path.join("/dvmm-filer2/datasets/PersonalityVideos/ECCV", "validationframes"))
train_framefolders = os.listdir(os.path.join("/dvmm-filer2/datasets/PersonalityVideos/ECCV", "trainframes"))

### train
for i in range(0, 6000):
    sourcepath = os.path.join(sourcefolder, "trainframes", train_framefolders[i])
    destpath = os.path.join(destfolder, "trainlandmarkAU")
    command = OpenfaceBase + '/FeatureExtraction' + ' -fdir ' + sourcepath + ' -2Dfp -aus -out_dir ' + destpath
    print(i, command)
    subprocess.call(command, shell = True)

### validation
for i in range(0, 2000):
    sourcepath = os.path.join(sourcefolder, "validationframes", validation_framefolders[i])
    destpath = os.path.join(destfolder, "validationlandmarkAU")
    command = OpenfaceBase + '/FeatureExtraction' + ' -fdir ' + sourcepath + ' -2Dfp -aus -out_dir ' + destpath
    print(i, command)
    subprocess.call(command, shell = True)
    
