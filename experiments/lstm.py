from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import keras
import keras.optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
import contrib.lstm.model_scripts.models
import contrib.utils
import pickle
import numpy as np


def build(config, dataset):
    experiment_dir = os.path.join(config.log_dir, config.experiment_name)
    weights_dir = os.path.join(experiment_dir, 'weights_' + config.experiment_name)

    class LSTMExperiment:
        def __init__(self):
            train_generator, validation_generator, test_generator = dataset.build(config)
            train_generator.batch_size = config.batch_size
            validation_generator.batch_size = config.eval_batch_size
            test_generator.batch_size = config.eval_batch_size
            self.train_generator = train_generator
            self.validation_generator = validation_generator
            self.test_generator = test_generator

        def run(self):
            # important to change the feature dimension size earlier it was 7168 and now it is 1024
            feature_size = 1024  # 7168
            # model = contrib.lstm.model_scripts.models.Sequence_decoder(config.num_neurons,
            #                                                            config.dropout,
            #                                                            feature_size,
            #                                                            dataset.NUM_CLASSES)

            model = contrib.lstm.model_scripts.models.Sequence_decoder_LSTM(config.num_neurons,
                                                                       config.dropout,
                                                                       feature_size,
                                                                       dataset.NUM_CLASSES)

            if config.mode == "test":
                # model.load_weights("/data/stars/user/sasharma/PKU_poseattnet/output/pku_rgb_cross_subject_lstm_final11/weights_pku_rgb_cross_subject_lstm_final11/epoch3.hdf5")
                # model.load_weights("/data/stars/user/sasharma/PKU_poseattnet/output/pku_rgb_cross_subject_gru_50epochs/weights_pku_rgb_cross_subject_gru_50epochs/epoch11.hdf5")  # 72.67 mAP 7x1024 features used
                # model.load_weights("/data/stars/user/sasharma/PKU_poseattnet/output/pku_rgb_cross_subject_64I3D_1024feat/weights_pku_rgb_cross_subject_64I3D_1024feat/epoch8.hdf5")  # 73.05 mAP for mean between 7 and 1024 features
                # model.load_weights("/data/stars/user/sasharma/PKU_poseattnet/output/pku_rgb_cross_subject_64I3D_1024feat_max/weights_pku_rgb_cross_subject_64I3D_1024feat_max/epoch3.hdf5")  # 72.07 mAP
                # model.load_weights("/data/stars/user/sasharma/PKU_poseattnet/output/pku_rgb_cross_subject_GRU32/weights_pku_rgb_cross_subject_GRU32/epoch7.hdf5")  # 80.26 mAP   for 32 frames check with epoch 8 once
                # model.load_weights("/data/stars/user/sasharma/PKU_poseattnet/output/pku_rgb_cross_subject_LSTM32/weights_pku_rgb_cross_subject_LSTM32/epoch10.hdf5")  # 81.17 mAP   for 32 frames check with epoch 8 once
                # model.load_weights("/data/stars/user/sasharma/PKU_poseattnet/output/pku_rgb_cross_subject_GRU16/weights_pku_rgb_cross_subject_GRU16/epoch6.hdf5")  # mAP  82.55 for 16 frames LSTM
                # model.load_weights("/data/stars/user/sasharma/PKU_poseattnet/output/pku_rgb_cross_subject_poseattnet_LSTM16/weights_pku_rgb_cross_subject_poseattnet_LSTM16/epoch8.hdf5")  # mAP  82.55 for 16 frames LSTM poseattnet
                # model.load_weights("/data/stars/user/sasharma/PKU_poseattnet/output/pku_rgb_cross_subject_poseattnet_nopretrainedNTU/weights_pku_rgb_cross_subject_poseattnet_nopretrainedNTU/epoch5.hdf5")  # mAP   for 16 frames LSTM poseattnet no pretrained on NTU
                # model.load_weights("/data/stars/user/sasharma/PKU_poseattnet/output/pku_rgb_cross_subject_poseattnet_pretrainedNTU/weights_pku_rgb_cross_subject_poseattnet_pretrainedNTU/epoch4.hdf5")  # mAP  85.31 for 16 frames LSTM poseattnet
                # model.load_weights("/data/stars/user/sasharma/PKU_poseattnet/output/pku_rgb_cross_subject_poseattnet_feature_LSTM10frames/weights_pku_rgb_cross_subject_poseattnet_feature_LSTM10frames/epoch3.hdf5")  # mAP  85.31 for 10 frames LSTM poseattnet

                # for I3D baseline for 10 frames input
                model.load_weights("/data/stars/user/sasharma/PKU_poseattnet/output/pku_rgb_cross_subject_I3D_feature_10frames_LSTM/weights_pku_rgb_cross_subject_I3D_feature_10frames_LSTM/epoch6.hdf5")  # mAP  85.31 for 10 frames LSTM poseattnet
                print("Checkpoint loaded successfully!")

            optimizer = keras.optimizers.Adam(lr=0.005, clipnorm=1)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            if config.use_predict == "False" and config.num_gpus > 1:
                parallel_model = keras.utils.multi_gpu_model(model, gpus=config.num_gpus)
                parallel_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            model_checkpoint = contrib.utils.CustomModelCheckpoint(model, os.path.join(weights_dir, 'epoch'))
            csv_logger = CSVLogger(os.path.join(experiment_dir, config.experiment_name + '.csv'))

            if config.class_weight == "True":
                class_weight = np.load('../data/class_weights.npy')
                print("Class weighting added to handle background videos")
            else:
                class_weight = None

            if config.mode == "train":
                print("Start training on I3D extracted features")
                parallel_model.fit_generator(
                    generator=self.train_generator,
                    validation_data=self.test_generator,
                    epochs=config.num_epochs,
                    callbacks=[csv_logger, model_checkpoint],
                    max_queue_size=48,
                    workers=multiprocessing.cpu_count() - 2,
                    use_multiprocessing=True,
                    class_weight=class_weight
                )
            elif config.mode == "test":
                print("Start test on GRU")
                """
                print(model.summary())
                features = model.evaluate_generator(
                    generator=self.test_generator,
                    workers=6,
                    use_multiprocessing=True
                 )
                print("features  ", features)
                """
                # """
                predictions = {}
                count = 0
                for x, y, videoname in self.test_generator:
                    # print("videoname ", videoname)
                    probs = model.predict(x)
                    print(model.evaluate(x, y))
                    predictions[videoname] = probs
                    count += 1
                    if count == 132:  # number of validation videos
                        break

                print("number of videos ", len(predictions.keys()))
                # pickle.dump(predictions, open('PKU_GRU_clsprobs_test_by_video_epoch11_'+str(feature_size)+'_poseattnet_LSTM_16_epoch3.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                # pickle.dump(predictions, open('PKU_GRU_clsprobs_'+str(feature_size)+'_poseattnet_nopretrainedNTU_epoch5_test.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                pickle.dump(predictions, open('PKU_LSTM_clsprobs_'+str(feature_size)+'_I3D_feature_10frames_epoch6_test.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
                # """

            # save model in json format
            # self.save_model_json(model)

        def save_model_json(self, model):
            model_json = model.to_json()
            with open(os.path.join(os.path.join(experiment_dir, config.experiment_name + '.json')), "w") as json_file:
                json_file.write(model_json)
            print("Saved model to disk")

    return LSTMExperiment()
