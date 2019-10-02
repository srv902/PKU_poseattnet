from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os
import datetime
import time
import tensorflow as tf
import data.factory
import experiments.i3d
import experiments.lstm
import debug


# Shared flags
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', dest='dataset_name', default=None, type=str)
parser.add_argument('--model', dest='model', default=None, type=str)
parser.add_argument('--num_epochs', dest='num_epochs', default=None, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=None, type=int)
parser.add_argument('--eval_batch_size', dest='eval_batch_size', default=None, type=int)
parser.add_argument('--experiment_name',
                    dest='experiment_name',
                    default='experiment_' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'),
                    type=str)
parser.add_argument('--log_dir', dest='log_dir', default=None, type=str)
parser.add_argument('--num_gpus', dest='num_gpus', default=2, type=int)
parser.add_argument('--gpus', dest='gpus', default="False", type=str)
parser.add_argument('--class_weight', dest='class_weight', default="False", type=str)

# Dahlia flags
parser.add_argument('--dahlia_train_path', dest='dahlia_train_path', default=None, type=str)
parser.add_argument('--dahlia_validation_path', dest='dahlia_validation_path', default=None, type=str)
parser.add_argument('--dahlia_test_path', dest='dahlia_test_path', default=None, type=str)
parser.add_argument('--dahlia_skeletons_dir', dest='dahlia_skeletons_dir', default=None, type=str)
parser.add_argument('--dahlia_modality', dest='dahlia_modality', default=None, type=str)


# PKU-MMD flags
parser.add_argument('--pku_mmd_split', dest='pku_mmd_split', default=None, type=str)
parser.add_argument('--pku_mmd_rgb_dir', dest='pku_mmd_rgb_dir', default=None, type=str)
parser.add_argument('--pku_mmd_skeletons_dir', dest='pku_mmd_skeletons_dir', default=None, type=str)
parser.add_argument('--pku_mmd_labels_dir', dest='pku_mmd_labels_dir', default=None, type=str)
parser.add_argument('--pku_mmd_splits_dir', dest='pku_mmd_splits_dir', default=None, type=str)
parser.add_argument('--pku_mmd_modality', dest='pku_mmd_modality', default=None, type=str)

# LSTM flags
parser.add_argument('--dropout', dest='dropout', default=0.5, type=float)
parser.add_argument('--time_steps', dest='time_steps', default=30, type=int)
parser.add_argument('--num_neurons', dest='num_neurons', default=512, type=int)
parser.add_argument('--data_size', dest='data_size', default=128, type=int)




FLAGS = parser.parse_args()

DATASET_NAME = FLAGS.dataset_name
MODEL = FLAGS.model
CW = FLAGS.class_weight
gpus = FLAGS.gpus

NUM_EPOCHS = FLAGS.num_epochs
BATCH_SIZE = FLAGS.batch_size
EVAL_BATCH_SIZE = FLAGS.eval_batch_size
EXPERIMENT_NAME = FLAGS.experiment_name
EXPERIMENT_DIR = os.path.join(FLAGS.log_dir, EXPERIMENT_NAME)
WEIGHTS_DIR = os.path.join(EXPERIMENT_DIR, 'weights_' + EXPERIMENT_NAME)

FLAGS.__setattr__('experiment_dir', EXPERIMENT_DIR)
FLAGS.__setattr__('weights_dir', WEIGHTS_DIR)


if not os.path.exists(EXPERIMENT_DIR):
    os.makedirs(EXPERIMENT_DIR)

if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)

dataset = data.factory.get_dataset(DATASET_NAME)

if MODEL == 'lstm':
    experiment = experiments.lstm.build(FLAGS, dataset)
elif MODEL == 'i3d':
    experiment = experiments.i3d.build(FLAGS, dataset, CW)
else:
    raise ValueError('Model %s not recognized' % MODEL)

experiment.run()
