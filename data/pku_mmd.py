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


IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_CLASSES = 51
DEFAULT_BATCH_SIZE = 32
STACK_SIZE = 64
STRIDE = 2

STEP = 30
SKELETON_FEATURE_SIZE = 150
NUM_JOINTS = 25
NUM_SUBJECTS = 2


def build(config):
    def _build_generator(samples):
        if config.pku_mmd_modality == 'skeletons':
            generator = PKUMMDSkeletons(samples, config)
        elif config.pku_mmd_modality == 'rgb':
            generator = PKUMMDRgb(samples, config)
        else:
            raise ValueError("Modality %s not recognized" % config.pku_mmd_modality)
        return generator
    train_split, validation_split, test_split = _gen_splits(config)
    train_generator = _build_generator(train_split)
    validation_generator = _build_generator(validation_split)
    test_generator = _build_generator(test_split)
    return train_generator, validation_generator, test_generator


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
        start_frame -= 1
        end_frame -= 1
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
            if count==0:                 #for the first video in the training list
               bckg.append(tuple([i[0]] + [0] + [1] + [i[2]-1]))
               count +=1
               last_video = tuple([i[0]]+[i[3]])
               continue
            if last_video[0] == i[0]:    #adding background for the same video
               bckg.append(tuple([i[0]] + [0] + [last_video[1]+1] + [i[2]-1]))
            else:                        #adding background when the video changes in the training list
               bckg.append(tuple([i[0]] + [0] + [1] + [i[2]-1]))
            last_video = tuple([i[0]]+[i[3]])
            count +=1
        return bckg

    train_split_actions = build_split(train_videos)
    background_split = add_bckg(train_split_actions)
    train_split = train_split_actions + background_split
    validation_split = build_split(validation_videos)
    test_split = build_split(test_videos)
    return train_split, validation_split, test_split


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
        selection = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self._get_data(selection)
        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.samples)

    @abstractmethod
    def _get_data(self, selection):
        pass


class PKUMMDRgb(PKUMMDBase):
    def __init__(self, samples, config):
        super(PKUMMDRgb, self).__init__(samples, config)

    def _get_data(self, selection):
        x = [self._get_video(sample) for sample in selection]
        x = np.array(x, np.float32)
        x /= 127.5
        x -= 1
        y = np.array([sample[1] for sample in selection]) #bckg is already 0
        y = keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
        return x, y

    def _get_video(self, sample):
        video_id, label, start_frame, end_frame = sample
        if start_frame > end_frame:
            print(video_id, label, start_frame, end_frame)
            return [np.random.random((IMG_WIDTH, IMG_HEIGHT, 3)) for i in range(64)]
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
        frames = [cv2.resize(cv2.imread(frame_path), (IMG_WIDTH, IMG_HEIGHT)) for frame_path in selected_paths]
        return frames


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
