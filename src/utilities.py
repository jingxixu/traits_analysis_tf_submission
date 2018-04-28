'''

----------------------------------------------------------------------
-- Some helpler functions to get a batch of videos, audio and ground
-- truth for training and validation

-- mostly by Jingxi Xu
-- landmark2im by Xuefeng Hu
----------------------------------------------------------------------

'''


import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import logging
import time
from joblib import Parallel, delayed

##############################################################################################

## Parameters to modify if running on different machines

train_target_file = '/home/jingxi/traits_analysis_tf/ECCV_gt/training_gt.csv'
validation_target_file = '/home/jingxi/traits_analysis_tf/ECCV_gt/validation_gt.csv'

train_frames_path = '/data/junting/ECCV/trainframes'
validation_frames_path = '/data/junting/ECCV/validationframes'

train_audiofeat_path = '/data/junting/ECCV/trainaudiofeat/'
validation_audiofeat_path = '/data/junting/ECCV/validationaudiofeat/'

train_ldau_path = '/data/junting/ECCV/trainldau/'
validation_ldau_path = '/data/junting/ECCV/validationldau/'

## 

##############################################################################################



# first load ground truth into a numpy file
train_gt_np = np.genfromtxt(train_target_file, delimiter=',', dtype=object)
validation_gt_np = np.genfromtxt(validation_target_file, delimiter=',', dtype=object)

# get a full list of mp4 names
train_mp4_names = train_gt_np[1:, 0]
validation_mp4_names = validation_gt_np[1:, 0]

# construct a dictionary for ground truth
train_gt = {}
validation_gt = {}
for i in range(1,train_gt_np.shape[0]):
    name = train_gt_np[i, 0]
    scores = train_gt_np[i, 1:].astype(float)
    train_gt[name] = scores
for i in range(1,validation_gt_np.shape[0]):
    name = validation_gt_np[i, 0]
    scores = validation_gt_np[i, 1:].astype(float)
    validation_gt[name] = scores

# get a batch of video, audio and ground truth for training
def get_next_batch(batch_size):
    batch = {}
    length = len(train_mp4_names)
    train_names=np.random.permutation(train_mp4_names)
    epoch_count = 0
    batch_count = 0
    start = 0
    while True:
        # randomly choose a list of mp4 names to consist the batch
        if start >= length or start + batch_size > length:
            train_names = np.random.permutation(train_mp4_names)
            epoch_count += 1
            start = 0
            batch_count = 0
        mp4_names = train_names[start: start + batch_size]
        batch_count += 1
        # get video batches -> batch_size * 6 * 112 * 112 * 3
        # video = np.zeros((batch_size, 6, 112, 112, 3))
        pre_time = time.time()
        vid_names = np.array(mp4_names)
        with Parallel(n_jobs=batch_size/4) as parallel:
            frames = parallel(delayed(load_vid_img)(vid_name)
                              for vid_name in vid_names)
            video = np.asarray(frames)
        print("epoch:{}, batch{}, load frames use: {}s".format(epoch_count, batch_count, time.time()-pre_time))
                # get audio batches -> batch_size * 6 * 68
        audio = np.zeros((batch_size, 6, 68))
        for batch_num, mp4 in enumerate(mp4_names):
            audiofeat_name = mp4+'.wav.csv'
            audio[batch_num] = np.genfromtxt(os.path.join(train_audiofeat_path, audiofeat_name), delimiter=',')

        # get ground truth -> batch_size * 6 * 5 (copied 5 times)
        gt = np.zeros((batch_size, 6, 5))
        for batch_num, mp4 in enumerate(mp4_names):
            gt[batch_num] = np.tile(np.array(train_gt[mp4]), (6, 1))

        batch['gt'] = gt
        batch['audio'] = audio
        batch['video'] = video
        start += batch_size
        yield batch

# get a batch of video audio and ground truth for validation
def get_next_validation_batch(batch_size):
    batch = {}
    length = len(validation_mp4_names)
    validation_names=np.random.permutation(validation_mp4_names)
    epoch_count = 0
    batch_count = 0
    start = 0
    while True:
        # randomly choose a list of mp4 names to consist the batch
        print(start)
        if start >= length or start + batch_size > length:
            break
        mp4_names = validation_names[start: start + batch_size]
        batch_count += 1
        # get video batches -> batch_size * 6 * 112 * 112 * 3
        # video = np.zeros((batch_size, 6, 112, 112, 3))
        pre_time = time.time()
        vid_names = np.array(mp4_names)
        with Parallel(n_jobs=batch_size/4) as parallel:
            frames = parallel(delayed(load_validation_vid_img)(vid_name)
                              for vid_name in vid_names)
            video = np.asarray(frames)
        print("epoch:{}, batch{}, load frames use: {}s".format(epoch_count, batch_count, time.time()-pre_time))
                # get audio batches -> batch_size * 6 * 68
        audio = np.zeros((batch_size, 6, 68))
        for batch_num, mp4 in enumerate(mp4_names):
            audiofeat_name = mp4+'.wav.csv'
            audio[batch_num] = np.genfromtxt(os.path.join(validation_audiofeat_path, audiofeat_name), delimiter=',')

        # get ground truth -> batch_size * 6 * 5 (copied 5 times)
        gt = np.zeros((batch_size, 5))
        for batch_num, mp4 in enumerate(mp4_names):
            gt[batch_num] = np.array(validation_gt[mp4])

        batch['gt'] = gt
        batch['audio'] = audio
        batch['video'] = video
        start += batch_size

        yield batch

# get batch function for the new model
def get_next_batch_multiscale(batch_size, frames_per_partition):
    batch = {}
    length = len(train_mp4_names)
    train_names=np.random.permutation(train_mp4_names)
    epoch_count = 0
    batch_count = 0
    start = 0
    while True:
        # randomly choose a list of mp4 names to consist the batch
        if start >= length or start + batch_size > length:
            train_names = np.random.permutation(train_mp4_names)
            epoch_count += 1
            start = 0
            batch_count = 0
        mp4_names = train_names[start: start + batch_size]
        batch_count += 1
        # get video batches -> batch_size * 6 * 112 * 112 * 3
        pre_time = time.time()
        vid_names = np.array(mp4_names)
        frames_per_partition_list = frames_per_partition * np.ones_like(mp4_names)
        with Parallel(n_jobs=batch_size/4) as parallel:
            video_and_sequence = parallel(delayed(load_vid_img_multiscale)(vid_name, frames_per_partition)
                              for vid_name, frames_per_partition in zip(vid_names, frames_per_partition_list))
            video_temp = np.asarray(video_and_sequence)[:, 0]
            video = np.zeros((batch_size, 6, frames_per_partition, 112, 112, 3))
            for i in range(batch_size):
                video[i] = video_temp[i]
            sequences_temp = np.asarray(video_and_sequence)[:, 1]
            sequences = np.zeros((batch_size, 6*frames_per_partition), dtype=int)
            for i in range(batch_size):
                sequences[i] = sequences_temp[i]
        print("epoch:{}, batch{}, load frames use: {}s".format(epoch_count, batch_count, time.time()-pre_time))
                # get audio batches -> batch_size * 6 * 68
        audio = np.zeros((batch_size, 6, 68))
        for batch_num, mp4 in enumerate(mp4_names):
            audiofeat_name = mp4+'.wav.csv'
            audio[batch_num] = np.genfromtxt(os.path.join(train_audiofeat_path, audiofeat_name), delimiter=',')

        # get ground truth -> batch_size * 5 
        gt = np.zeros((batch_size, 5))
        for batch_num, mp4 in enumerate(mp4_names):
            gt[batch_num] = np.array(train_gt[mp4])
            
        # get landmark -> batch_size * 112 * 112
        # get action units -> batch_size * 17
        landmark = np.zeros((batch_size, 6, 8, 136))
        au = np.zeros((batch_size, 6, 8, 17))
        for batch_num, mp4 in enumerate(mp4_names):
            ldau_name = mp4.replace('.mp4', '.csv')
            ldau = np.genfromtxt(os.path.join(train_ldau_path, ldau_name), delimiter=',')[1:, 5:]
            print("mp4 : ", mp4)
            print("sequences: ",batch_num," ; ", sequences[batch_num])
            landmark[batch_num] = ldau[sequences[batch_num]][:,:136].reshape(6, 8, 136)
            au[batch_num] = ldau[sequences[batch_num]][:, 136:153].reshape(6, 8, 17)
        landmark = landmark2im(landmark)
        
        batch['gt'] = gt
        batch['audio'] = audio
        batch['video'] = video
        batch['landmark'] = landmark
        batch['au'] = au
        start += batch_size
        yield batch


######### NOT IN USE
# get the set for validation
def get_our_validation_set():
    print("Producing Validation Set")
    validation_set = {}
    
    # to construct validation set, use all videos
    mp4_names = validation_mp4_names

    # get video batches -> 2000 * 6 * 112 * 112 * 3
    video = np.zeros((2000, 6, 112, 112, 3))
    for batch_num, mp4 in enumerate(mp4_names):            
        all_frames = os.listdir(os.path.join(validation_frames_path, mp4.replace('.mp4', '')))
        num_frames = len(all_frames) # int
        interval = num_frames/6
        for i in range(0, 6):
            index = i*interval + np.random.randint(1, interval+1)
            frame_name = 'frame_det_00_%06d.png' % (index)
            frame = mpimg.imread(os.path.join(validation_frames_path, mp4.replace('.mp4', ''), frame_name))
            video[batch_num, i] = frame

    # get audio batches -> 2000 * 6 * 68
    audio = np.zeros((2000, 6, 68))
    for batch_num, mp4 in enumerate(mp4_names):
        audiofeat_name = mp4+'.wav.csv'
        audio[batch_num] = np.genfromtxt(os.path.join(validation_audiofeat_path, audiofeat_name), delimiter=',')

    # get ground truth -> 2000 * 6 * 5 (copied 5 times)
    gt = np.zeros((2000, 6, 5))
    for batch_num, mp4 in enumerate(mp4_names):
        gt[batch_num] = np.tile(np.array(validation_gt[mp4]), (6, 1))

    validation_set['gt'] = gt
    validation_set['audio'] = audio
    validation_set['video'] = video
    
    np.save("/data/junting/ECCV/validation_set/run1_gt.npy", gt)
    np.save("/data/junting/ECCV/validation_set/run1_audio.npy", audio)
    np.save("/data/junting/ECCV/validation_set/run1_video.npy", video)
    print("Validation Set Saved!")

    return validation_set

def load_vid_img(mp4):
    all_frames = os.listdir(os.path.join(train_frames_path, mp4.replace('.mp4', '')))
    num_frames = len(all_frames)  # int
    interval = num_frames / 6
    frames=[]
    for i in range(0, 6):
        index = i * interval + np.random.randint(1, interval + 1)
        frame_name = 'frame_det_00_%06d.png' % (index)
        frame = mpimg.imread(os.path.join(train_frames_path, mp4.replace('.mp4', ''), frame_name))
        frames.append(frame)
    return frames

def load_validation_vid_img(mp4):
    all_frames = os.listdir(os.path.join(validation_frames_path, mp4.replace('.mp4', '')))
    num_frames = len(all_frames)  # int
    interval = num_frames / 6
    frames=[]
    for i in range(0, 6):
        index = i * interval + np.random.randint(1, interval + 1)
        frame_name = 'frame_det_00_%06d.png' % (index)
        frame = mpimg.imread(os.path.join(validation_frames_path, mp4.replace('.mp4', ''), frame_name))
        frames.append(frame)
    return frames

## for nultiscale model
def load_vid_img_multiscale(mp4, frames_per_partition):
    sequence = []
    ## frames_per_partition should be defines in the caller
    all_frames = os.listdir(os.path.join(train_frames_path, mp4.replace('.mp4', '')))
    num_frames = len(all_frames)  # int
    interval = num_frames / 6
    # for a video -> 6 * 8 * 112 * 112 * 3
    video=np.zeros((6, 8, 112, 112, 3))
    
    # for each partition -> 8 * 112 * 112 * 3
    for n_partition in range(0, 6):
        frames=np.zeros((8, 112, 112, 3))
        indices = np.zeros(frames_per_partition)
        interval_frame = interval/frames_per_partition
        # get index for each frame of a partition
        for n_frame in range(frames_per_partition):
            indices[n_frame] = n_partition * interval + n_frame * interval_frame + np.random.randint(1, interval_frame + 1)
        sequence += list(indices)
        # for each frame in this partition -> 112 * 112 *3
        for j, index in enumerate(indices):
            frame_name = 'frame_det_00_%06d.png' % (index)
            # print frame_name
            frame = mpimg.imread(os.path.join(train_frames_path, mp4.replace('.mp4', ''), frame_name))
            frames[j] = frame
        video[n_partition] = frames
    return [video, np.asarray(sequence)]


## for multiscale model
def load_validation_vid_img_multiscale(mp4, frames_per_partition):
    pass


# transfer landmark array into image
def landmark2im(landmark):
    partition_num = 6
    frames_per_partition = 8
    landmarksize = 68
    #check dimension
    batch_size, nb_partition, nb_frames, nb_landmark = np.shape(landmark)
    if(nb_partition != partition_num or nb_frames!= frames_per_partition  or nb_landmark != 2 * landmarksize):
        raise ValueError("Wrong input dimension: [None, 6, 8, 136] required")
    #process data
    landmark = landmark.astype(int)
    landmark[landmark>=112]=111
    landmark[landmark<=0]=0
    im = np.zeros([batch_size, partition_num, frames_per_partition, 112, 112])
    landmark_coordinate = np.reshape(landmark, [batch_size, partition_num, frames_per_partition, landmarksize, 2],'F')
    landmark_coordinate = landmark_coordinate.astype(int)
    coords = np.array([
                        [sample_id, 
                         partition_id, 
                         frame_id, 
                         landmark_coordinate[sample_id, partition_id, frame_id, feats_id, 0], 
                         landmark_coordinate[sample_id, partition_id, frame_id, feats_id, 1]] 
                        for sample_id in range(batch_size)
                        for partition_id in range(partition_num)
                        for frame_id in range(frames_per_partition) 
                        for feats_id in range(landmarksize) 
                    ])
    im[coords[:,0],coords[:,1],coords[:,2],coords[:,3],coords[:,4]] = 1
    return im


# to create the parent directory (recursively if the fiolder structure is not existing already)
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if os.path.isdir(path):
            pass
        else:
            raise
