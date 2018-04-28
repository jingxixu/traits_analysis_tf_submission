'''

----------------------------------------------------------------------
-- convert bmp cropped face frames to png frames for validation data

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

filepath = '/dvmm-filer2/datasets/PersonalityVideos/Interview/'
source_directory = os.path.join(filepath, 'validationframes')

count = 0
allfiles = os.listdir(source_directory)
for i in range(len(allfiles)):
    file = allfiles[i]
    bmp_directory = os.path.join(source_directory, file)
    dest_directory = os.path.join('/dvmm-filer2/datasets/PersonalityVideos/Interview/validationframes_images', file)
    mkdir_p(dest_directory)
    for bmp in os.listdir(bmp_directory):
        img = Image.open(os.path.join(bmp_directory, bmp))
        img.save(os.path.join(dest_directory, bmp.replace('.bmp', '.png')), 'png')
    count+=1
    print(count)
