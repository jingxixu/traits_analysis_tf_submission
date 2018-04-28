
'''

----------------------------------------------------------------------
-- convert bmp cropped face frames to png frames for training data

-- Jingxi Xu
----------------------------------------------------------------------

'''


import os
from PIL import Image

def mkdir_p(path):
    try: 
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if os.path.isdir(path):
            pass
        else:
            raise

filepath = os.getcwd()
source_directory = os.path.join(filepath, 'trainframes')

count = 0
allfiles = os.listdir(source_directory)
for i in range(0,6000):
    file = allfiles[i]
    bmp_directory = os.path.join(source_directory, file, file+'_aligned')
    dest_directory = os.path.join('/dvmm-filer2/datasets/PersonalityVideos/ECCV/trainframes_images', file)
    mkdir_p(dest_directory)
    for bmp in os.listdir(bmp_directory):
        img = Image.open(os.path.join(bmp_directory, bmp))
        img.save(os.path.join(dest_directory, bmp.replace('.bmp', '.png')), 'png')
    count+=1
    print(count)
