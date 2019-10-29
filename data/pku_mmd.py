from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
import glob
import os
import random
import cv2
import numpy as np
import keras
from operator import itemgetter
import pandas as pd
import pickle
import math
import contrib.utils


IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CLASSES = 52    # 51
DEFAULT_BATCH_SIZE = 32
STACK_SIZE = 10  # 32
STRIDE = 2

STEP = 10  # number of skeletons to consider
SKELETON_FEATURE_SIZE = 150
NUM_JOINTS = 25
NUM_SUBJECTS = 2


def build(config):
    def _build_feat_generator(path, mode):
        if config.pku_mmd_modality == 'skeletons':
            generator = PKUMMDFeat(config)
        elif config.pku_mmd_modality == 'rgb':
            generator = PKUMMDFeat(path, config, mode)
        else:
            raise ValueError("Modality %s not recognized" % config.pku_mmd_modality)
        return generator

    def _build_generator(samples, mode):
        if config.pku_mmd_modality == 'skeletons':
            generator = PKUMMDSkeletons(samples, config)
        elif config.pku_mmd_modality == 'rgb' and mode == "train":
            generator = PKUMMDRgb(samples, config)
        elif config.pku_mmd_modality == 'rgb' and mode == "test":
            generator = PKUMMDRgb_Test(samples, config)
        else:
            raise ValueError("Modality %s not recognized" % config.pku_mmd_modality)
        return generator

    def _build_poseattnet_generator(samples):
        generator = PKUMMDRgb(samples, config)
        return generator

    if config.model == 'i3d':
        train_split, validation_split, test_split = _gen_splits32(config)   # changed gen_splits to consider for 32 frame scenarios.. all functions wrt 64 frame evaluation are removed..
        train_generator = _build_generator(train_split, "train")
        validation_generator = _build_generator(validation_split, "train")
        test_generator = _build_generator(test_split, "test")
    elif config.model == "lstm":
        train_generator = _build_feat_generator(path='/data/stars/user/sasharma/PKU_poseattnet/output', mode="train")
        validation_generator = _build_feat_generator(path='/data/stars/user/sasharma/PKU_poseattnet/output', mode="test")
        test_generator = _build_feat_generator(path='/data/stars/user/sasharma/PKU_poseattnet/output', mode="test")
    elif config.model == "poseattnet":
        train_split, validation_split, test_split = _gen_splits32(config)
        train_generator = _build_poseattnet_generator(train_split)
        validation_generator = _build_poseattnet_generator(validation_split)
        test_generator = _build_poseattnet_generator(test_split)

    return train_generator, validation_generator, test_generator


def _gen_splits32(config):
    train_videos = []
    validation_videos = []
    split_path = os.path.join(config.pku_mmd_splits_dir, config.pku_mmd_split + '.txt')
    all_videos = [os.path.splitext(os.path.basename(v))[0]
                  for v in glob.glob(os.path.join(config.pku_mmd_rgb_dir, '*'))]
    with open(split_path, 'r') as f:
        for line in f:
            if 'Training' in line:
                train_videos = [s.strip() for s in next(f).strip().split(',') if s.strip() in all_videos][82:]
            elif 'Validation' in line:
                validation_videos = [s.strip() for s in next(f).strip().split(',') if s.strip() in all_videos]
    print("train_videos", sorted(train_videos))
    print("validation_videos", sorted(validation_videos))
    test_videos = [v for v in all_videos if ((v not in train_videos) and (v not in validation_videos))]

    def process_row(row):
        label, start_frame, end_frame, _ = [int(i) for i in row.split(',')]
        return [label, start_frame, end_frame]

    def build_split(videos):
        split = [
            tuple([video] + process_row(row))
            for video, labels_path in zip(videos, [os.path.join(config.pku_mmd_labels_dir, v + '.txt') for v in videos])
            for row in open(labels_path)
        ]
        return split

    def add_backg(train_split):
        count = 0
        bckg = []
        for i in train_split:
            if count == 0:  # for the first video in the training list
                bckg.append(tuple([i[0]] + [0] + [1] + [i[2] - 1]))
                count += 1
                last_video = tuple([i[0]] + [i[3]])
                continue
            if last_video[0] == i[0]:  # adding background for the same video
                if (i[2] - last_video[1]) > 10:
                    bckg.append(tuple([i[0]] + [0] + [last_video[1] + 1] + [i[2] - 1]))
            else:  # adding background when the video changes in the training list
                if i[2] != 0:
                    bckg.append(tuple([i[0]] + [0] + [1] + [i[2] - 1]))

            last_video = tuple([i[0]] + [i[3]])
            count += 1
        return bckg

    # create 20 frame overlapping windows of 64 frames..
    def create_ovr_window(test_split):
        stride = 44  # overlap of 20 frames are considered before the end of the buffer
        final_split = []
        for data in test_split:
            tmp_split = []
            video, action, start_frame, end_frame = data[0], data[1], data[2], data[3]
            for idx in range(start_frame, end_frame - 64 + 1, stride):
                tmp_split.append((video, action, idx, idx + 64))
                # print("tmp_split")
                # print(tmp_split)
            final_split.extend(tmp_split)

        final_split = sorted(final_split, key=itemgetter(0))

        return final_split

    def generate_split(config, sub):
        stride = STACK_SIZE  # 44
        test_split = []
        path = config.pku_mmd_rgb_dir
        subjects = sub  # os.listdir(path)

        for f in subjects:
            frames = os.listdir(os.path.join(path, f))
            numframes = len(frames)
            # print(f)
            labels = pd.read_csv(os.path.join(config.pku_mmd_labels_dir, f + '.txt'), header=None)
            labels = labels.sort_values(by=labels.columns[1])  # sort the gt segments for sanity

            assert numframes > 0
            tmp = []
            for j in range(1, numframes, stride):  # changed from numframes - 65 to only numframes
                start, end = j, j + stride
                if end > numframes:
                    continue
                # check for overlaps in ground truth segments with the current segment
                overlap = labels[np.logical_not(np.logical_or(labels.iloc[:, 1] >= end, labels.iloc[:, 2] <= start))]

                # if multiple overlaps found, assign the label with the label of segment which max overlaps with the current segment
                if overlap.shape[0] >= 1:
                    time_in_wins = (np.minimum(end, overlap.iloc[:, 2]) - np.maximum(start, overlap.iloc[:, 1]))
                    highfreqid = time_in_wins.idxmax()
                    label = labels.iloc[highfreqid, 0]
                else:
                    label = 0
                tmp.append((f, label, start, end))

            test_split.extend(tmp)
        return test_split

    def get_split(train_split):
        final_train_split = []
        df = pd.DataFrame(columns=['video-id', 'label', 't-start', 't-end'])
        for i in range(len(train_split)):
            tmp = {'video-id': train_split[i][0], 'label': train_split[i][1], 't-start': train_split[i][2],
                   't-end': train_split[i][3]}
            df = df.append(tmp, ignore_index=True)

        print("Number of entries in the df ", df.shape)
        # print("dataframe ")
        # print(df)

        train_groups = df.groupby('video-id')
        train_videos = df['video-id'].unique()
        for i in train_videos:
            vidgroup = train_groups.get_group(i)
            endidx = vidgroup['t-end'].max()
            for idx in range(1, endidx - 65, 64):
                start, end = idx, idx + 64
                label = vidgroup[np.logical_not(np.logical_or(vidgroup.iloc[:, 2] >= end, vidgroup.iloc[:, 3] <= start))]['label'].max()
                final_train_split.append((i, label, start, end))

        # print("train split new ")
        # print(final_train_split)
        # create 64 frame chunk and assign label..
        return final_train_split

    # train_split_actions = build_split(train_videos)
    # background_split = add_backg(train_split_actions)

    # add background
    # train_split = train_split_actions + background_split

    # sort the train split
    # train_split = sorted(train_split, key=itemgetter(0, 2))

    # create 64 frame training split with corresponding labels

    # print("train split")
    # print(train_split)

    # correct way of doing should be below
    # train_split = train_split_actions
    # train_split = generate_split(config, train_videos)

    # print(len(train_split))
    # train_split = get_split(train_split)

    # pickle.dump(train_split, open('../output/train_split_no_overlap.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    # train_split = pickle.load(open('../output/train_split_no_overlap.pkl', 'rb'))
    # print("train_split")
    # print(train_split)
    # print(test_split)

    # print("new train split")
    # print(train_split)

    # no background
    # train_split = train_split_actions

    # added for 32 or 16 frame case one time run, then use pickle file for loading splits ********
    # train_split = generate_split(config, train_videos)

    # print("train_split before num frames check ", len(train_split))
    # train_split = [i for i in train_split if i[3] - i[2] >= 10]
    # print("train split after num frames check ", len(train_split))

    # pickle.dump(train_split, open('../output/train_split_32frames.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    # train_split = pickle.load(open('../output/train_split_32frames.pkl', 'rb'))  # ************

    # pickle.dump(train_split, open('../output/train_split_16frames.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    # train_split = pickle.load(open('../output/train_split_16frames.pkl', 'rb'))  # ************

    # pickle.dump(train_split, open('../output/train_split_10frames.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    train_split = pickle.load(open('../output/train_split_10frames.pkl', 'rb'))  # ************

    # validation_split = build_split(validation_videos)
    # validation_split = generate_split(config, validation_videos)  # one time run, then use pickle file for loading splits *******

    # print("val before num frames check ", len(validation_split))
    # validation_split = [i for i in validation_split if i[3] - i[2] >= 10]
    # print("val before num frames check ", len(validation_split))

    # load below for extraction for 32 frames
    # pickle.dump(validation_split, open('../output/validation_split_32frames.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    # validation_split = pickle.load(open('../output/validation_split_32frames.pkl', 'rb'))  # ***********

    # load below for extraction for 16 frames
    # pickle.dump(validation_split, open('../output/validation_split_16frames.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    # validation_split = pickle.load(open('../output/validation_split_16frames.pkl', 'rb'))  # ***********

    # load below for extraction for 10 frames
    # pickle.dump(validation_split, open('../output/validation_split_10frames.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    validation_split = pickle.load(open('../output/validation_split_10frames.pkl', 'rb'))  # ***********

    # create overlapping windows of 64 frames for test
    # the output from the model needs to be organized into (label, start, end, confidence, video_name)
    # test_split = build_split(test_videos)
    # test_bkg_split = add_backg(test_split)
    # test_split = test_split + test_bkg_split

    # test_split = [i for i in test_split if i[3] - i[2] > 10]

    # test_dict = create_sample_dict(test_split)
    # print(" Before Test split ", test_split)
    # test_split = create_ovr_window(test_split)
    # print(" After Test split ", test_split)
    # test_dict = create_sample_dict(test_split)
    # print("************Test dict ***********")
    # print(test_dict)
    # print("Number of test samples ", len(test_split))

    if config.feature_extract == "False":
        train_split_actions = build_split(train_videos)
        background_split = add_backg(train_split_actions)
        train_split = train_split_actions + background_split
        print("train_split before num frames check ", len(train_split))
        train_split = [i for i in train_split if i[3] - i[2] > 10]
        print("train split after num frames check ", len(train_split))
        validation_split = build_split(validation_videos)
        print("val before num frames check ", len(validation_split))
        validation_split = [i for i in validation_split if i[3] - i[2] > 10]
        print("val before num frames check ", len(validation_split))
    else:
        train_split = pickle.load(open('../output/train_split_10frames.pkl', 'rb'))
        validation_split = pickle.load(open('../output/validation_split_10frames.pkl', 'rb'))
        # train_split = pickle.load(open('../output/train_split_16frames.pkl', 'rb'))
        # validation_split = pickle.load(open('../output/validation_split_16frames.pkl', 'rb'))

    # just to keep test and validation split as same
    test_split = validation_split

    return train_split, validation_split, test_split


def _gen_splits(config):
    train_videos = []
    validation_videos = []
    split_path = os.path.join(config.pku_mmd_splits_dir, config.pku_mmd_split + '.txt')
    all_videos = [os.path.splitext(os.path.basename(v))[0]
                  for v in glob.glob(os.path.join(config.pku_mmd_rgb_dir, '*'))]
    with open(split_path, 'r') as f:
        for line in f:
            if 'Training' in line:
                train_videos = [s.strip() for s in next(f).strip().split(',') if s.strip() in all_videos][82:]
            elif 'Validation' in line:
                validation_videos = [s.strip() for s in next(f).strip().split(',') if s.strip() in all_videos]
    print("train_videos", sorted(train_videos))
    print("validation_videos", sorted(validation_videos))
    test_videos = [v for v in all_videos if ((v not in train_videos) and (v not in validation_videos))]

    def process_row(row):
        label, start_frame, end_frame, _ = [int(i) for i in row.split(',')]
        return [label, start_frame, end_frame]

    def build_split(videos):
        split = [
            tuple([video] + process_row(row))
            for video, labels_path in zip(videos, [os.path.join(config.pku_mmd_labels_dir, v + '.txt') for v in videos])
            for row in open(labels_path)
        ]
        return split

    def add_backg(train_split):
        count = 0
        bckg = []
        for i in train_split:
            if count == 0:                 #for the first video in the training list
               bckg.append(tuple([i[0]] + [0] + [1] + [i[2]-1]))
               count += 1
               last_video = tuple([i[0]]+[i[3]])
               continue
            if last_video[0] == i[0]:                   #adding background for the same video
                if (i[2] - last_video[1]) > 10:
                    bckg.append(tuple([i[0]] + [0] + [last_video[1]+1] + [i[2]-1]))
            else:                        #adding background when the video changes in the training list
               if i[2] != 0:
                   bckg.append(tuple([i[0]] + [0] + [1] + [i[2]-1]))

            last_video = tuple([i[0]]+[i[3]])
            count +=1
        return bckg

    # create 20 frame overlapping windows of 64 frames..
    def create_ovr_window(test_split):
        stride = 44  # overlap of 20 frames are considered before the end of the buffer
        final_split = []
        for data in test_split:
            tmp_split = []
            video, action, start_frame, end_frame = data[0], data[1], data[2], data[3]
            for idx in range(start_frame, end_frame-64+1, stride):
                tmp_split.append((video, action, idx, idx+64))
                #print("tmp_split")
                #print(tmp_split)
            final_split.extend(tmp_split)

        final_split = sorted(final_split, key=itemgetter(0))

        return final_split

    def generate_test_split(config, sub):
        stride = 64  # 44
        test_split = []
        path = config.pku_mmd_rgb_dir
        subjects = sub   # os.listdir(path)

        for f in subjects:
            frames = os.listdir(os.path.join(path, f))
            numframes = len(frames)
            labels = pd.read_csv(os.path.join(config.pku_mmd_labels_dir, f+'.txt'), header=None)

            assert numframes > 0
            tmp = []
            for j in range(1, numframes, stride):   # changed from numframes - 65 to only numframes
                start, end = j, j+64
                if end > numframes:
                    continue
                label = labels[np.logical_not(np.logical_or(labels.iloc[:, 1] >= end, labels.iloc[:, 2] <= start))].iloc[:, 0].max()
                # print("label type ", type(label))
                if math.isnan(label):
                    label = 0
                tmp.append((f, label, start, end))
            test_split.extend(tmp)

        return test_split

    # creates a dictionary which contains all the background and foreground samples to be later used for creating windows
    def create_sample_dict(test_split):
        test_dict = {}
        for i in test_split:
            if i[0] not in test_dict.keys():
                test_dict[i[0]] = []
                test_dict[i[0]] = [(i[0], i[1], i[2], i[3])]
            else:
                test_dict[i[0]].append((i[0], i[1], i[2], i[3]))

        for key in test_dict:
            test_dict[key] = sorted(test_dict[key], key=itemgetter(2))

        return test_dict


    def get_split(train_split):
        final_train_split = []
        df = pd.DataFrame(columns=['video-id', 'label', 't-start', 't-end'])
        for i in range(len(train_split)):
            tmp = {'video-id': train_split[i][0], 'label': train_split[i][1], 't-start': train_split[i][2], 't-end': train_split[i][3]}
            df = df.append(tmp, ignore_index=True)

        print("Number of entries in the df ", df.shape)
        # print("dataframe ")
        # print(df)

        train_groups = df.groupby('video-id')
        train_videos = df['video-id'].unique()
        for i in train_videos:
            vidgroup = train_groups.get_group(i)
            endidx = vidgroup['t-end'].max()
            for idx in range(1, endidx-65, 64):
                start, end = idx, idx+64
                label = vidgroup[np.logical_not(np.logical_or(vidgroup.iloc[:, 2] >= end, vidgroup.iloc[:, 3] <= start))]['label'].max()
                final_train_split.append((i, label, start, end))

        # print("train split new ")
        # print(final_train_split)
        # create 64 frame chunk and assign label..
        return final_train_split

    # train_split_actions = build_split(train_videos)
    # background_split = add_backg(train_split_actions)

    # add background
    # train_split = train_split_actions + background_split

    # sort the train split
    # train_split = sorted(train_split, key=itemgetter(0, 2))

    # create 64 frame training split with corresponding labels

    # print("train split")
    # print(train_split)

    # correct way of doing should be below
    # train_split = train_split_actions
    # train_split = generate_test_split(config, train_videos)

    # print(len(train_split))
    # train_split = get_split(train_split)

    # pickle.dump(train_split, open('../output/train_split_no_overlap.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    train_split = pickle.load(open('../output/train_split_no_overlap.pkl', 'rb'))
    # print("train_split")
    # print(train_split)
    # print(test_split)

    # print("new train split")
    # print(train_split)

    # no background
    # train_split = train_split_actions

    print("train_split before num frames check ", len(train_split))
    train_split = [i for i in train_split if i[3] - i[2] > 10]
    print("train split after num frames check ", len(train_split))

    validation_split = build_split(validation_videos)
    print("val before num frames check ", len(validation_split))
    validation_split = [i for i in validation_split if i[3] - i[2] > 10]
    print("val before num frames check ", len(validation_split))

    # create overlapping windows of 64 frames for test
    # the output from the model needs to be organized into (label, start, end, confidence, video_name)
    # test_split = build_split(test_videos)
    # test_bkg_split = add_backg(test_split)
    # test_split = test_split + test_bkg_split

    # test_split = [i for i in test_split if i[3] - i[2] > 10]

    # test_dict = create_sample_dict(test_split)
    # print(" Before Test split ", test_split)
    # test_split = create_ovr_window(test_split)
    # print(" After Test split ", test_split)
    # test_dict = create_sample_dict(test_split)
    # print("************Test dict ***********")
    # print(test_dict)
    # print("Number of test samples ", len(test_split))

    # ************************************** below for test_split
    if config.model != 'lstm':
        test_split = generate_test_split(config, validation_videos)
        print("Number of test split entries ", len(test_split))
        pickle.dump(test_split, open('../output/test_split_no_overlap_' + config.model + '_new.pkl', 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        print("test split saved as pickle ")
        # test_split = test_split[:-2]
    else:
        test_split = generate_test_split(config, validation_videos)
        # print(test_split)
        test_split = test_split[:-2]
        # print("test_split")
        # print(test_split)
        # print("length of test split ", len(test_split))
        # pickle.dump(test_split, open('../output/test_split_no_overlap_' + config.model + '.pkl', 'wb'),
        #             pickle.HIGHEST_PROTOCOL)
        return test_split

    return train_split, validation_split, test_split


class PKUMMDFeat(keras.utils.Sequence):
    def __init__(self, path, config, mode):
        self.batch_size = 1
        self.mode = mode
        self.config = config
        if STACK_SIZE == 64:
            ofeat = 7
        elif STACK_SIZE == 32:
            ofeat = 3
        elif STACK_SIZE == 16:
            ofeat = 1
        elif STACK_SIZE == 10:
            ofeat = 1
        if self.config.use_predict == "False" and self.mode == "train":
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKU_train_features.pkl'), 'rb')) for 64
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKU_features_no_overlap_train32.pkl'), 'rb'))  # for 32
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKU_features_no_overlap_train16.pkl'), 'rb'))  # for 16
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKUGCNN_features_no_overlap_train16.pkl'), 'rb'))  # for 16 Final_PKUGCNN_features_no_overlap_train16.pkl
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKUGCNN_features_no_overlap_train16_no_pretrainedNTU.pkl'), 'rb'))  # for 16 Final_PKUGCNN_features_no_overlap_train16.pkl no pretrained
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKUGCNN_features_no_overlap_train16_pretrainedNTU.pkl'), 'rb'))  # for 16 Final_PKUGCNN_features_no_overlap_train16.pkl pretrained on NTU
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKUGCNN_features_no_overlap_train10.pkl'), 'rb')) # final test on 10 frame input
            self.data = pickle.load(open(os.path.join(path, 'Final_PKUI3D_features_no_overlap_train10.pkl'), 'rb')) # final test on 10 frame input for I3D baseline

            self.data = np.reshape(self.data, (self.data.shape[0], ofeat, 1024))  # added to reduce 7 x 1024 to 1024
            self.data = np.mean(self.data, axis=1)  # added due to above reason
            # self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_train_labels_interval.pkl'), 'rb'))  # for 64
            # self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_train_labels_interval_modified32.pkl'), 'rb'))  # for 32
            # self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_train_labels_interval_modified16.pkl'), 'rb'))  # for 16
            self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_train_labels_interval_modified10.pkl'), 'rb'))  # for 10 frames
            self.videonames = list(self.videos.keys())
            self.numvideos = len(self.videos.keys())
        elif self.config.use_predict == "False" and self.mode == "test":
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKU_test_features.pkl'), 'rb'))
            # self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_test_labels_interval.pkl'), 'rb'))
            # self.data = pickle.load(open(os.path.join(path, 'PKU_features_no_overlap_test_new.pkl'), 'rb'))  # for 64
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKU_features_no_overlap_test32.pkl'), 'rb'))  # for 32
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKU_features_no_overlap_test16.pkl'), 'rb'))  # for 16

            # self.data = pickle.load(open(os.path.join(path, 'Final_PKUGCNN_features_no_overlap_test16.pkl'), 'rb'))  # for 16 Final_PKUGCNN_features_no_overlap_test16.pkl  test on epoch 3 of I3D no pretrained
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKUGCNN_features_no_overlap_test16_no_pretrainedNTU.pkl'), 'rb'))  # for 16 Final_PKUGCNN_features_no_overlap_test16.pkl no pretrained
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKUGCNN_features_no_overlap_test16_pretrainedNTU.pkl'), 'rb'))  # for 16 Final_PKUGCNN_features_no_overlap_test16.pkl pretrained on NTU
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKUGCNN_features_no_overlap_test10.pkl'), 'rb'))  # for 10 frame input
            self.data = pickle.load(open(os.path.join(path, 'Final_PKUI3D_features_no_overlap_test10.pkl'), 'rb'))  # for 10 frame input for I3D baseline
            self.data = np.reshape(self.data, (self.data.shape[0], ofeat, 1024))  # added to reduce 7 x 1024 to 1024
            self.data = np.mean(self.data, axis=1)  # added due to above reason

            # self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_train_labels_interval.pkl'), 'rb'))
            # self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_test_labels_interval_modified.pkl'), 'rb'))   # for 64
            # self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_test_labels_interval_modified32.pkl'), 'rb'))   # for 32
            # self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_test_labels_interval_modified16.pkl'), 'rb'))   # for 16
            self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_test_labels_interval_modified10.pkl'), 'rb'))   # for 10
            self.videonames = list(self.videos.keys())
            self.numvideos = len(self.videos.keys())
        elif self.config.use_predict == "True" and self.config.mode == "test":
            # self.data = pickle.load(open(os.path.join(path, 'PKU_features_no_overlap_test_new.pkl'), 'rb'))  # for 64
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKU_features_no_overlap_test32.pkl'), 'rb'))  # for 32
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKU_features_no_overlap_test16.pkl'), 'rb'))  # for 16
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKUGCNN_features_no_overlap_test16.pkl'), 'rb'))  # for 16 for epoch 3 of I3D no pretrained
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKUGCNN_features_no_overlap_test16_no_pretrainedNTU.pkl'), 'rb'))  # for 16 frames I3D best epoch 19 no pretrained on NTU
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKUGCNN_features_no_overlap_test16_pretrainedNTU.pkl'), 'rb'))  # for 16 frames I3D best epoch 19 pretrained on NTU
            # self.data = pickle.load(open(os.path.join(path, 'Final_PKUGCNN_features_no_overlap_test10.pkl'), 'rb'))  # for 10 frame input
            self.data = pickle.load(open(os.path.join(path, 'Final_PKUI3D_features_no_overlap_test10.pkl'), 'rb'))  # for 10 frame input for I3D baseline
            self.data = np.reshape(self.data, (self.data.shape[0], ofeat, 1024))  # added to reduce 7 x 1024 to 1024
            self.data = np.mean(self.data, axis=1)  # added due to above reason
            # self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_train_labels_interval.pkl'), 'rb'))
            # self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_test_labels_interval_modified.pkl'), 'rb'))  # for 64
            # self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_test_labels_interval_modified32.pkl'), 'rb'))  # for 32
            # self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_test_labels_interval_modified16.pkl'), 'rb'))  # for 16
            self.videos = pickle.load(open(os.path.join(path, 'Final_PKU_test_labels_interval_modified10.pkl'), 'rb'))  # for 16
            self.videonames = list(self.videos.keys())
            self.numvideos = len(self.videos.keys())

    def __len__(self):
        return int(self.numvideos/self.batch_size)

    def __getitem__(self, videoid):
        if self.config.use_predict == "False":
            videoname = self.videonames[videoid]
            start, end = self.videos[videoname]['start'], self.videos[videoname]['end']
            x = self.data[start:end+1]
            x = np.reshape(x, (1, x.shape[0], x.shape[1]))
            # print("x shape ", x.shape)

            # prepare target for training
            if self.mode == "test":
                target = self.videos[videoname]['label']
                # print("in test to get generator")
            elif self.mode == "train":
                target = self.videos[videoname]['label']
                # print("in train to get generator")

            # y = np.array([sample[1] for sample in selection])  # bckg is already 0
            y = keras.utils.to_categorical(target, num_classes=NUM_CLASSES)
            y = np.reshape(y, (1, y.shape[0], y.shape[1]))
            return x, y
        else:
            videoname = self.videonames[videoid]
            start, end = self.videos[videoname]['start'], self.videos[videoname]['end']
            x = self.data[start:end + 1]
            x = np.reshape(x, (1, x.shape[0], x.shape[1]))

            target = self.videos[videoname]['label']
            y = keras.utils.to_categorical(target, num_classes=NUM_CLASSES)
            y = np.reshape(y, (1, y.shape[0], y.shape[1]))
            # x = self.data[videoid]
            # x = np.reshape(x, (1, 1, 7168))
            # print("shape of x ", x.shape)
            return x, y, videoname

    def on_epoch_end(self):
        if self.config.mode == "train":
            np.random.shuffle(self.videonames)
        else:
            pass

class PKUMMDBase(keras.utils.Sequence):
    def __init__(self, samples, config):
        self._batch_size = 2
        self.samples = samples
        self.config = config

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.samples) / self._batch_size))

    def __getitem__(self, index):
        # print("index  s ", index)
        selection = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
        # print("selection ", selection)
        if self.config.model == "poseattnet":
            alpha, beta, SYM_NORM = 5, 1, True
            A = contrib.utils.compute_adjacency(alpha, beta)
            A = np.repeat(A, self.batch_size, axis=0)
            A = np.reshape(A, [self.batch_size, A.shape[1], A.shape[1]])
            graph_conv_filters = contrib.utils.preprocess_adj_tensor_with_identity(A, SYM_NORM)
            graph_conv = graph_conv_filters[0:self.batch_size]
            y_reg = np.zeros([self.batch_size])
            x, y, X = self._get_data(selection)
            return [X[:, 0, :, :], X[:, 1, :, :], X[:, 2, :, :], X[:, 3, :, :], X[:, 4, :, :], X[:, 5, :, :], X[:, 6, :, :],
                X[:, 7, :, :], X[:, 8, :, :], X[:, 9, :, :], X, graph_conv, x], [y, y_reg]
        else:
            x, y = self._get_data(selection)
            return x, y

    def on_epoch_end(self):
        if self.config.feature_extract == "False":
            np.random.shuffle(self.samples)
        else:
            pass

    @abstractmethod
    def _get_data(self, selection):
        pass

    @abstractmethod
    def _get_skeleton(self, selection):
        pass

class PKUMMDRgb_Test(PKUMMDBase):
    def __init__(self, samples, config):
        super(PKUMMDRgb_Test, self).__init__(samples, config)

    def _get_data(self, selection):
        # print("selection ", selection)
        x = [self._get_video(sample) for sample in selection]
        x = np.array(x, np.float32)
        x /= 127.5
        x -= 1
        # print("x shape ", x.shape)

        if self.config.mode == "train" and self.config.feature_extract == "False":
            y = np.array([sample[1] for sample in selection])   # bckg is already 0
            y = keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
            return x, y
        elif self.config.mode == "test":
            return x, 0
        elif self.config.mode == "train" and self.config.feature_extract == "True":
            return x, 0
        else:
            print("Incorrect mode specified at DataLoader class!!")

    def _get_video(self, sample):
        video_id, label, start_frame, end_frame = sample

        frame_paths = sorted(glob.glob(os.path.join(self.config.pku_mmd_rgb_dir, video_id, '*')))[start_frame:end_frame]
        selected_paths = []

        if self.config.mode == "train":
            if len(selected_paths) > (STACK_SIZE * STRIDE):
                start = random.randint(0, len(frame_paths) - STACK_SIZE * STRIDE)
                selected_paths.extend([frame_paths[i] for i in range(start, (start + STACK_SIZE * STRIDE), STRIDE)])
            elif len(frame_paths) < STACK_SIZE:
                selected_paths.extend(frame_paths)
                while len(selected_paths) < STACK_SIZE:
                    selected_paths.extend(frame_paths)
                selected_paths = selected_paths[:STACK_SIZE]
            else:
                start = random.randint(0, len(frame_paths) - STACK_SIZE)
                selected_paths.extend([frame_paths[i] for i in range(start, (start + STACK_SIZE))])
        elif self.config.mode == "test":
            selected_paths = frame_paths
        else:
            print("Incorrect mode specified!!")

        selected_paths.sort()
        frames = []
        # print("selected paths ")
        # print(selected_paths)
        # print("length of the files ", len(selected_paths))
        assert len(selected_paths) == STACK_SIZE
        for frame_path in selected_paths:
            if os.path.isfile(frame_path):
                tmp = cv2.imread(frame_path)
                tmp = cv2.resize(tmp, (IMG_WIDTH, IMG_HEIGHT))
                frames.append(tmp)
            else:
                print("Invalid image file!!")

        # print("video id ", video_id)
        # print("frames size", len(frames))
        return frames


class PKUMMDRgb(PKUMMDBase):
    def __init__(self, samples, config):
        super(PKUMMDRgb, self).__init__(samples, config)

    def _get_data(self, selection):
        # print("selection ", selection)
        x = [self._get_video(sample) for sample in selection]
        x = np.array(x, np.float32)
        x /= 127.5
        x -= 1
        y = np.array([sample[1] for sample in selection])   # bckg is already 0
        y = keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

        if self.config.model == "poseattnet":
            skeleton = self._get_skeleton(selection)
            return x, y, skeleton

        return x, y

    def _get_video(self, sample):
        video_id, label, start_frame, end_frame = sample
        # print("sample in get video ", sample)
        try:
            if start_frame > end_frame:
                #print(" " + str(video_id) + " " +str(label) +" " + str(start_frame) + " " + str(end_frame))
                return [np.random.random((IMG_WIDTH, IMG_HEIGHT, 3)) for i in range(STACK_SIZE)]
            frame_paths = sorted(glob.glob(os.path.join(self.config.pku_mmd_rgb_dir, video_id, '*')))[start_frame:end_frame]
            selected_paths = []

            if len(selected_paths) > (STACK_SIZE * STRIDE):
                start = random.randint(0, len(frame_paths) - STACK_SIZE * STRIDE)
                selected_paths.extend([frame_paths[i] for i in range(start, (start + STACK_SIZE * STRIDE), STRIDE)])
            elif len(frame_paths) < STACK_SIZE:
                selected_paths.extend(frame_paths)
                while len(selected_paths) < STACK_SIZE:
                    selected_paths.extend(frame_paths)
                selected_paths = selected_paths[:STACK_SIZE]
            else:
                start = random.randint(0, len(frame_paths) - STACK_SIZE)
                selected_paths.extend([frame_paths[i] for i in range(start, (start + STACK_SIZE))])

            selected_paths.sort()
        except:
            print("Error occurred!!!!")

        frames = []
        for frame_path in selected_paths:
            try:
                tmp = cv2.imread(frame_path)
                tmp = cv2.resize(tmp, (IMG_WIDTH, IMG_HEIGHT))
                frames.append(tmp)
            except:
                print("some errors in the frame path ", frame_path)

        # print("video id ", video_id)
        # print("frames size", len(frames))
        return frames

    def _get_skeleton(self, selection):
        x = np.empty((self._batch_size, STEP, SKELETON_FEATURE_SIZE))

        for i, sample in enumerate(selection):
            video_id, label, start_frame, end_frame = sample
            # print("video id ", video_id, label, start_frame, end_frame)
            skeleton_path = os.path.join(self.config.pku_mmd_skeletons_dir, video_id + '.txt')
            # print("skeleton path ", skeleton_path)
            unpadded_file = np.loadtxt(skeleton_path)[start_frame:end_frame]
            origin1 = unpadded_file[0, 3:6]  # Extract hip of the first frame
            origin2 = unpadded_file[0, 78:81]
            [row, col] = unpadded_file.shape
            origin = np.concatenate([np.tile(origin1, NUM_JOINTS), np.tile(origin2, NUM_JOINTS)])
            unpadded_file = unpadded_file - origin  # translation
            extra_frames = (len(unpadded_file) % STEP)
            l = 0
            if len(unpadded_file) < STEP:
                extra_frames = STEP - len(unpadded_file)
                l = 1
            if extra_frames < int(STEP / 2) & l == 0:
                padded_file = unpadded_file[0:len(unpadded_file) - extra_frames, :]
            else:
                [row, col] = unpadded_file.shape
                alpha = int(len(unpadded_file) / STEP) + 1
                req_pad = np.zeros(((alpha * STEP) - row, col))
                padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, STEP)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0, STEP):
                c = np.random.choice(col, 1)
                sampled_file.append(splitted_file[k, c, :])
            sampled_file = np.asarray(sampled_file)
            x[i, :] = np.squeeze(sampled_file)

        x = x[:, :, :75]
        x = np.reshape(x, [self.batch_size, STEP, 25, 3])
        return x


class PKUMMDSkeletons(PKUMMDBase):
    def __init__(self, samples, config):
        super(PKUMMDSkeletons, self).__init__(samples, config)

    def _get_data(self, selection):
        x = np.empty((self._batch_size, STEP, SKELETON_FEATURE_SIZE))

        for i, sample in enumerate(selection):
            video_id, label, start_frame, end_frame = sample
            skeleton_path = os.path.join(self.config.pku_mmd_skeletons_dir, video_id + '.txt')
            unpadded_file = np.loadtxt(skeleton_path)
            origin1 = unpadded_file[0, 3:6]  # Extract hip of the first frame
            origin2 = unpadded_file[0, 78:81]
            [row, col] = unpadded_file.shape
            origin = np.concatenate([np.tile(origin1, NUM_JOINTS), np.tile(origin2, NUM_JOINTS)])
            unpadded_file = unpadded_file - origin  # translation
            extra_frames = (len(unpadded_file) % STEP)
            l = 0
            if len(unpadded_file) < STEP:
                extra_frames = STEP - len(unpadded_file)
                l = 1
            if extra_frames < int(STEP / 2) & l == 0:
                padded_file = unpadded_file[0:len(unpadded_file) - extra_frames, :]
            else:
                [row, col] = unpadded_file.shape
                alpha = int(len(unpadded_file) / STEP) + 1
                req_pad = np.zeros(((alpha * STEP) - row, col))
                padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, STEP)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0, STEP):
                c = np.random.choice(col, 1)
                sampled_file.append(splitted_file[k, c, :])
            sampled_file = np.asarray(sampled_file)
            x[i, :] = np.squeeze(sampled_file)

        y = np.array([s[1] for s in selection]) - 1
        return x, keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

