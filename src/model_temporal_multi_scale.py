'''

----------------------------------------------------------------------
-- Deep Temporal Multi-scale Net

-- Qiangeng Xu
----------------------------------------------------------------------

'''

import tensorflow as tf
from ops import *

# qx2128

class Qiangeng_Net(object):
    def __init__(self):
        self.frames_per_partition = 8
        self.num_au = 17

    def create_model(self):
        '''
            This function defined the model flow of  Deep Temporal Multi-scale Net in initial tensorflow graph building
            The feed_dict send in np array of ground_truth label, audio vectors, cropped face frame sequences, landmark sequences
            and au code vectors.
            The model in the end will output the prediction scores of 5 personality traits.
        '''
        self.ground_truth = tf.placeholder(tf.float32, shape=[None, 5], name="ground_truth")
        self.audio_pl = tf.placeholder(tf.float32, shape=[None, 6, 68], name="audio_pl")
        self.video_pl = tf.placeholder(tf.float32, shape=[None, 6,
                          self.frames_per_partition, 112, 112, 3], name="video_pl")
        self.landmark_pl = tf.placeholder(tf.float32, shape=[None, 6,
                          self.frames_per_partition, 112, 112], name="landmark_pl")
        self.au_pl = tf.placeholder(tf.float32, shape=[None, 6,
                          self.frames_per_partition, self.num_au], name="au_pl")
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        # -------------------- Audio Branch ----------------------
        # -- -1 * 6 x 68 --> -1 * 6 x 32
        audio_branch = tf.reshape(self.audio_pl,[-1, 68])
        audio_branch = linear(audio_branch, 32, "audio_fc1")

        # -------------------- Video Branch ----------------------
        # -- -1 * 6 * 8 * 112x112x(3+1)  --> -1 * 8 * 112x112x4
        visual_branch = tf.concat([self.video_pl, tf.expand_dims(self.landmark_pl, axis=5)], axis=5)
        # -- -1 * 6 * 8 * 112x112x4  --> -1 * 112x112x4
        visual_branch = tf.reshape(visual_branch, [-1, 112, 112, 4])

        # -- -1 * 112x112x4  --> -1 * 108x108x16
        visual_branch = conv2d(visual_branch, 16, 5, 5, name="vd_conv1", padding="VALID")
        visual_branch = relu(batch_norm(visual_branch, "vd_conv1_bn", train=self.is_train, reuse=False))
        # -- -1 * 108x108x16 --> -1 * 54x54x16
        print("visual_branch.get_shape().as_list()",visual_branch.get_shape().as_list())
        visual_branch = MaxPooling(visual_branch, 2, stride=2)

        # -- -1 * 54x54x16  --> -1 * 48x48x16
        visual_branch = conv2d(visual_branch, 16, 7, 7, name="vd_conv2", padding="VALID")
        visual_branch = relu(batch_norm(visual_branch, "vd_conv2_bn", train=self.is_train, reuse=False))
        # -- -1 * 48x48x16 --> -1 * 24x24x16
        visual_branch = MaxPooling(visual_branch, 2, stride=2)

        # reshape to 8 frames per partition -1 * 24x24x16 ---> -1 * 8 * 24x24x16
        visual_branch = tf.reshape(visual_branch, [-1, self.frames_per_partition, 24, 24, 16])

        # scale 1: 2 frames and sum up

        scale1_visual = self.get_scale1_features(visual_branch)

        # scale 2: 4 frames and sum up

        scale2_visual = self.get_scale2_features(visual_branch)

        # scale 3: 8 frames and sum up

        scale3_visual = self.get_scale3_features(visual_branch)
        # -1 * 96
        visual_features = tf.concat([scale1_visual, scale2_visual, scale3_visual], 1)

        # -------------------- AU code Branch ----------------------

        # -- -1 * 6 x 17 * 6 --> -1 * (17 * 6)
        AU_branch = tf.reshape(self.au_pl, [-1, self.frames_per_partition * self.num_au])
        # -- -1 x 17 * 6 --> -1 * (32)
        AU_branch = linear(AU_branch, 32, "au_fc1")

        # -------------------- Synergy Branch ----------------------
        # -1 * (32+96+32) ----> -1 * 160
        frame_features = tf.concat([audio_branch, AU_branch, visual_features], 1)
        # frame_features = tf.nn.dropout(frame_features, 1-0.2)
        frame_features = tf.reshape(frame_features, [-1, 6, 160])
        for_cell = tf.nn.rnn_cell.LSTMCell(num_units = 160, num_proj = 128, state_is_tuple=True)
        # [batch_size, 6, 128]
        frame_features, state = tf.nn.dynamic_rnn(for_cell, frame_features, dtype=frame_features.dtype)
        frame_features = tf.reshape(frame_features, [-1, 128])
        # frame_features = tf.nn.dropout(frame_features, 1-0.2)
        # -1 * 16
        frame_features = relu(batch_norm(linear(frame_features, 16, "fea_fc1"), "fea_fc1_bn", train=self.is_train, reuse=False))
        # -1 * (6*16)
        frame_features = tf.reshape(frame_features, [-1, 6 * 16])
        frame_features = tf.sigmoid(linear(frame_features, 5, "fea_fc2"))
        self.frame_features = tf.reshape(frame_features, [-1, 5])


    def get_scale1_features(self, visual_branch):
        '''
             This function defined the sub branch of visual information.
             The 3d convolution would summarize every 2 frames and
             return one feature vector for each partition
        '''
        # -- -1 * 8 * 24x24x16  --> -1 * 4 * 20x20x16
        visual_branch = conv3d(visual_branch, 16,
                               k_d=2, k_h=5, k_w=5, s_d=2, s_h=1, s_w=1, name="scale1_conv3d_1", padding='VALID')
        visual_branch = relu(batch_norm(visual_branch, "scale1_conv3d_1_bn", train=self.is_train, reuse=False))

        # -- -1 * 4 * 20x20x16  --> -1 * 4 * 16x16x16
        visual_branch = conv3d(visual_branch, 16,
                               k_d=1, k_h=5, k_w=5, s_d=1, s_h=1, s_w=1, name="scale1_conv3d_2", padding='VALID')
        visual_branch = relu(batch_norm(visual_branch, "scale1_conv3d_2_bn", train=self.is_train, reuse=False))
        # -- -1 * 4 * 16x16x16 --> -1 * 4 * 8x8x16
        visual_branch = tf.reshape(visual_branch, [-1, 16, 16, 16])
        visual_branch = MaxPooling(visual_branch, 2)
        visual_branch = tf.reshape(visual_branch, [-1, 4, 8, 8, 16])
        # -- -1 * 4 * 8x8x16  --> -1 * 4 * 4x4x8
        visual_branch = conv3d(visual_branch, 8,
                               k_d=1, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name="scale1_conv3d_3", padding='SAME')
        visual_branch = relu(batch_norm(visual_branch, "scale1_conv3d_3_bn", train=self.is_train, reuse=False))

        # -1 * 4 * 4x4x8 -> -1 * 128
        visual_branch = tf.reshape(visual_branch, [-1, 4 * 4 * 8])
        # -1 x 128 --> -1 x 8 --> -1 * 32
        visual_branch = tf.reshape(relu(batch_norm(
            linear(visual_branch, 8, "scale1_vd_fc1"), "scale1_vd_fc1_bn", train=self.is_train, reuse=False)),[-1, 32])

        return visual_branch

    def get_scale2_features(self, visual_branch):
        '''
             This function defined the sub branch of visual information.
             The 3d convolution would summarize every 4 frames and
             return one feature vector for each partition
        '''
        # -- -1 * 8 * 24x24x16  --> -1 * 4 * 20x20x16
        visual_branch = conv3d(visual_branch, 16,
                               k_d=2, k_h=5, k_w=5, s_d=2, s_h=1, s_w=1, name="scale2_conv3d_1", padding='VALID')
        visual_branch = relu(batch_norm(visual_branch, "scale2_conv3d_1_bn", train=self.is_train, reuse=False))

        # -- -1 * 4 * 20x20x16  --> -1 * 2 * 16x16x16
        visual_branch = conv3d(visual_branch, 16,
                               k_d=2, k_h=5, k_w=5, s_d=2, s_h=1, s_w=1, name="scale2_conv3d_2", padding='VALID')
        visual_branch = relu(batch_norm(visual_branch, "scale2_conv3d_2_bn", train=self.is_train, reuse=False))
        # -- -1 * 2 * 16x16x16 --> -1 * 2 * 8x8x16
        visual_branch = tf.reshape(visual_branch, [-1, 16, 16, 16])
        visual_branch = MaxPooling(visual_branch, 2)
        visual_branch = tf.reshape(visual_branch, [-1, 2, 8, 8, 16])

        # -- -1 * 2 * 8x8x16  --> -1 * 2 * 4x4x8
        visual_branch = conv3d(visual_branch, 8,
                               k_d=1, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name="scale2_conv3d_3", padding='SAME')
        visual_branch = relu(batch_norm(visual_branch, "scale2_conv3d_3_bn", train=self.is_train, reuse=False))

        # -1 * 2 * 4x4x8 -> -1 * 128
        visual_branch = tf.reshape(visual_branch, [-1, 4 * 4 * 8])
        # -1 x 128 --> -1 x 16 --> -1 * 32
        visual_branch = tf.reshape(relu(batch_norm(linear(
            visual_branch, 16, "scale2_vd_fc1"), "scale2_vd_fc1_bn", train=self.is_train, reuse=False)), [-1, 32])

        return visual_branch

    def get_scale3_features(self, visual_branch):
        '''
             This function defined the sub branch of visual information.
             The 3d convolution would summarize every 8 frames and
             return one feature vector for each partition
        '''
        # -- -1 * 8 * 24x24x16  --> -1 * 4 * 20x20x16
        visual_branch = conv3d(visual_branch, 16,
                               k_d=2, k_h=5, k_w=5, s_d=2, s_h=1, s_w=1, name="scale3_conv3d_1", padding='VALID')
        visual_branch = relu(batch_norm(visual_branch, "scale3_conv3d_1_bn", train=self.is_train, reuse=False))

        # -- -1 * 4 * 20x20x16  --> -1 * 2 * 16x16x16
        visual_branch = conv3d(visual_branch, 16,
                               k_d=2, k_h=5, k_w=5, s_d=2, s_h=1, s_w=1, name="scale3_conv3d_2", padding='VALID')
        visual_branch = relu(batch_norm(visual_branch, "scale3_conv3d_2_bn", train=self.is_train, reuse=False))
        # -- -1 * 2 * 16x16x16 --> -1 * 2 * 8x8x16
        visual_branch = tf.reshape(visual_branch, [-1, 16, 16, 16])
        visual_branch = MaxPooling(visual_branch, 2)
        visual_branch = tf.reshape(visual_branch, [-1, 2, 8, 8, 16])

        # -- -1 * 2 * 8x8x16  --> -1 * 1 * 4x4x8
        visual_branch = conv3d(visual_branch, 8,
                               k_d=2, k_h=3, k_w=3, s_d=2, s_h=2, s_w=2, name="scale3_conv3d_3", padding='SAME')
        visual_branch = relu(batch_norm(visual_branch, "scale3_conv3d_3_bn", train=self.is_train, reuse=False))

        # -1 * 1 * 4x4x8 -> -1 * 128
        visual_branch = tf.reshape(visual_branch, [-1, 4 * 4 * 8])
        # -1 x 128 --> -1 x 32 --> -1 * 6 * 32
        visual_branch = relu(batch_norm(linear(
            visual_branch, 32, "scale3_vd_fc1"), "scale3_vd_fc1_bn", train=self.is_train, reuse=False))

        return visual_branch

