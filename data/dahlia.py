from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from abc import abstractmethod
import numpy as np
import pandas as pd
import xml.etree.ElementTree
import os
import glob
import keras.utils
from random import sample, randint, shuffle
import cv2

# tf.flags.DEFINE_string('skeletons_dir', 'Directory where the .npz files containing the joints are stored', None)
# FLAGS = tf.flags.FLAGS


def build(config):
    def get_samples(path):
        if path is None:
            return []
        df = pd.read_csv(path, header=None)
        samples = [list(i[1]) for i in df.iterrows()]
        return samples

    def build_generator(samples):
        if config.dahlia_modality == 'skeletons':
            generator = DahliaSkeletons(samples, config)
        elif config.dahlia_modality == 'rgb':
            generator = DahliaRgb(samples, config)
        else:
            raise ValueError("Modality %s not recognized" % config.dahlia_modality)
        return generator

    train_generator = build_generator(get_samples(config.dahlia_train_path))
    validation_generator = build_generator(get_samples(config.dahlia_validation_path))
    test_generator = build_generator(get_samples(config.dahlia_test_path))
    validation_generator.batch_size = config.eval_batch_size
    test_generator.batch_size = config.eval_batch_size
    return train_generator, validation_generator, test_generator


NUM_CLASSES = 8
NUM_JOINTS = 25
# SKELETONS_DIR = '/data/stars/user/rriverag/dahlia/skeletons/proc'

LABELS = {
    'neutral': 0,
    'cooking': 1,
    'laying_table': 2,
    'eating': 3,
    'clearing_table': 4,
    'washing_dishes': 5,
    'housework': 6,
    'working': 7
}


class DahliaBase(keras.utils.Sequence):
    def __init__(self, samples, config):
        self._batch_size = config.batch_size
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


class DahliaRgb(DahliaBase):
    def __init__(self, samples, config):
        super(DahliaRgb, self).__init__(samples, config)
        self.stack_size = 64
        self.num_classes = 8
        self.stride = 2

    @staticmethod
    def _name_to_int(name):
        return LABELS[name]

    def _get_data(self, selection):
        x = [self._get_video(s) for s in selection]
        x = np.array(x, np.float32)
        x /= 127.5
        x -= 1
        y = np.array([LABELS[s[1]] for s in selection])
        y = keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
        return x, y

    def _get_video(self, bundle):
        frames_dir, class_name, start_frame, end_frame = bundle
        images = glob.glob(frames_dir + "/*")
        images.sort()
        images = images[int(start_frame):int(end_frame)]
        files = []
        if len(images) > (self.stack_size * self.stride):
            start = randint(0, len(images) - self.stack_size * self.stride)
            files.extend([images[i] for i in range(start, (start + self.stack_size * self.stride), self.stride)])
        elif len(images) < self.stack_size:
            files.extend(images)
            while len(files) < self.stack_size:
                files.extend(images)
            files = files[:self.stack_size]
        else:
            start = randint(0, len(images) - self.stack_size)
            files.extend([images[i] for i in range(start, (start + self.stack_size))])

        files.sort()

        arr = []
        for i in files:
            if os.path.isfile(i):
                arr.append(cv2.resize(cv2.imread(i), (224, 224)))
            else:
                arr.append(arr[-1])

        return arr


class DahliaSkeletons(DahliaBase):

    def __init__(self, samples, config):
        super(DahliaSkeletons, self).__init__(samples, config)
        self.step = 30
        self.dim = 75

    def _get_data(self, selection):
        x = np.empty((self.batch_size, self.step, self.dim))

        for i, bundle in enumerate(selection):
            main_id = bundle[0].split('/')[-2]
            skeleton_path = os.path.join(self.config.dahlia_skeletons_dir,
                                         '{}-{}-{}-{}.npz'.format(main_id,
                                                                  bundle[1],
                                                                  *[str(int(i)) for i in bundle[2:]]))
            unpadded_file = np.load(skeleton_path)['arr_0']
            origin = unpadded_file[0, 3:6]  # Extract hip of the first frame
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, NUM_JOINTS))  # making equal dimension
            unpadded_file = unpadded_file - origin  # translation
            extra_frames = (len(unpadded_file) % self.step)
            l = 0
            if len(unpadded_file) < self.step:
                extra_frames = self.step - len(unpadded_file)
                l = 1
            if extra_frames < int(self.step / 2) & l == 0:
                padded_file = unpadded_file[0:len(unpadded_file) - extra_frames, :]
            else:
                [row, col] = unpadded_file.shape
                alpha = int(len(unpadded_file) / self.step) + 1
                req_pad = np.zeros(((alpha * self.step) - row, col))
                padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0, self.step):
                c = np.random.choice(col, 1)
                sampled_file.append(splitted_file[k, c, :])
            sampled_file = np.asarray(sampled_file)
            x[i, :] = np.squeeze(sampled_file)

        y = np.array([LABELS[s[1]] for s in selection])
        return x, keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
