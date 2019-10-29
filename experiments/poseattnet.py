##############################
#
# PoseAttNet code reorganized for PKU by Saurav Sharma
# Taken from PoseAttNet NTU code by Srijan Das
#
##############################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import pickle
seed = 8
np.random.seed(seed)
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

import keras
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
from keras.utils import multi_gpu_model
from keras.models import Model
import multiprocessing
import contrib.utils
import contrib.poseattnet.embedding_gcnn_att_model

def build(config, dataset):
    experiment_dir = os.path.join(config.log_dir, config.experiment_name)
    weights_dir = os.path.join(experiment_dir, 'weights_' + config.experiment_name)

    class PoseAttNet:
        def __init__(self):
            print("PoseAttNet experiment")
            print("Number of classes ", dataset.NUM_CLASSES)
            train_generator, validation_generator, test_generator = dataset.build(config)
            train_generator.batch_size = config.batch_size
            validation_generator.batch_size = config.eval_batch_size
            test_generator.batch_size = config.eval_batch_size
            self.train_generator = train_generator
            self.validation_generator = validation_generator
            self.test_generator = test_generator
            print('train', train_generator, len(train_generator.samples), 'batch_size', train_generator.batch_size)
            print('validation', validation_generator, len(validation_generator.samples), 'batch_size', test_generator.batch_size)
            print('test', test_generator, len(test_generator.samples), 'batch_size', test_generator.batch_size)

        def run(self):
            losses = {
                "action_output": "categorical_crossentropy",
                "embed_output": "mean_squared_error",
            }
            lossWeights = {"action_output": 0.99, "embed_output": 0.01}
            optimizer = SGD(lr=0.01, momentum=0.9)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
            alpha, beta, num_features, num_nodes, num_filters, SYM_NORM, n_dropout = 5, 2, 3, 25, 2, True, 0.3
            A = contrib.utils.compute_adjacency(alpha, beta)
            A = np.repeat(A, config.batch_size, axis=0)
            A = np.reshape(A, [config.batch_size, A.shape[1], A.shape[1]])
            graph_conv_filters = contrib.utils.preprocess_adj_tensor_with_identity(A, SYM_NORM)

            if config.class_weight == "True":
                print("Class weighting will be performed to handle background videos")
                # class_weight = np.load('/data/stars/user/sdas/PKU-MMD/scripts/data/class_weights.npy')
                class_weight_action = np.load('../data/class_weights.npy')
                class_weight_embed = [1, 1]
                class_weight = {}
                class_weight['action_output'] = class_weight_action
                class_weight['embed_output'] = class_weight_embed
            else:
                class_weight = None

            # define poseattnet model here
            model = contrib.poseattnet.embedding_gcnn_att_model.embed_model_spatio_temporal_gcnn(config.n_neuron, config.timesteps, num_nodes, num_features,
                                                     graph_conv_filters.shape[1], graph_conv_filters.shape[2], num_filters, dataset.NUM_CLASSES, n_dropout, config.protocol)


            model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer, metrics=['accuracy'])

            # ntu weights location /data/stars/user/sdas/PhD_work/CVPR20/deployment_code/epoch_30.hdf5 - NTU-60 pre-trained model
            if config.use_ntu_weights == "True":
                model_ntu = contrib.poseattnet.embedding_gcnn_att_model.embed_model_spatio_temporal_gcnn(
                    config.n_neuron,
                    config.timesteps,
                    num_nodes,
                    num_features,
                    graph_conv_filters.shape[
                        1],
                    graph_conv_filters.shape[
                        2], num_filters,
                    60,
                    n_dropout,
                    config.protocol)
                model_ntu.load_weights('/data/stars/user/sdas/PhD_work/CVPR20/deployment_code/weights_NTU_60_t16/epoch_30.hdf5')
                model_ntu.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer, metrics=['accuracy'])
                for l_m, l_lh in zip(model.layers[0: len(model.layers)-7], model_ntu.layers[0: len(model.layers)-7]):
                    l_m.set_weights(l_lh.get_weights())
                    l_m.trainable = True
                print("NTU weights loaded successfully!!")

            # for extracting features from the poseattnet model for input to LSTM model 16 frames are considered for evaluation..
            if config.feature_extract == "True":
                # model.load_weights('../output/pku_rgb_cross_subject_poseattnet/weights_pku_rgb_cross_subject_poseattnet/epoch3.hdf5')
                # model.load_weights('../output/pku_rgb_cross_subject_poseattnet_epoch50/weights_pku_rgb_cross_subject_poseattnet_epoch50/epoch19.hdf5') # only I3d gcn no pretrained NTU
                # model.load_weights('../output/pku_rgb_cross_subject_poseattnet_NTU_pretraiend/weights_pku_rgb_cross_subject_poseattnet_NTU_pretraiend/epoch31.hdf5')  # only I3d gcn no pretrained NTU
                model.load_weights('../output/pku_rgb_cross_subject_poseattnet_10frames/weights_pku_rgb_cross_subject_poseattnet_10frames/epoch19.hdf5')  # only I3d gcn 10 frames
                model = Model(inputs=model.input, outputs=model.get_layer('global_avg_poolsecond').output)
                print("****************Checkpoint loaded successfully*********************")

            if config.num_gpus > 1:
                if config.feature_extract == "False":
                    parallel_model = keras.utils.multi_gpu_model(model, gpus=config.num_gpus)
                    parallel_model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer, metrics=['accuracy'])
                    model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer, metrics=['accuracy'])
                else:
                    parallel_model = keras.utils.multi_gpu_model(model, gpus=config.num_gpus)
                    parallel_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            # define model checkpoint
            # model_checkpoint = contrib.utils.CustomModelCheckpoint(model, './weights_' + config.model + '/epoch_')
            # model_checkpoint = contrib.utils.CustomModelCheckpoint(model, os.path.join(config.weights_dir, 'epoch'))
            model_checkpoint = contrib.utils.CustomModelCheckpoint(model, os.path.join(weights_dir, 'epoch'))
            csvlogger = CSVLogger(os.path.join(experiment_dir, config.experiment_name + '.csv'))

            if config.feature_extract == "False" and config.mode == "train":
                parallel_model.fit_generator(generator=self.train_generator,
                                             validation_data=self.validation_generator,
                                             use_multiprocessing=True,
                                             max_queue_size=48,
                                             epochs=config.num_epochs,
                                             callbacks=[csvlogger, reduce_lr, model_checkpoint],
                                             workers=multiprocessing.cpu_count() - 2, class_weight=class_weight)
            elif config.feature_extract == "True" and config.mode == "train":
                print("Start feature extracting on 16 frame train samples ")
                features = parallel_model.predict_generator(
                    generator=self.train_generator,
                    workers=multiprocessing.cpu_count() - 2,
                    use_multiprocessing=True,
                    max_queue_size=48,
                    verbose=1
                )
                print("PoseAttNet train output features shape ", features.shape)
                pickle.dump(features, open('PKU_poseattnet_feat_train10frames.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
            elif config.feature_extract == "True" and config.mode == "test":
                print("Start feature extracting on 16 frame test samples ")
                features = parallel_model.predict_generator(
                    generator=self.test_generator,
                    workers=multiprocessing.cpu_count() - 2,
                    use_multiprocessing=True,
                    max_queue_size=48,
                    verbose=1
                )
                print("PoseAttNet test output features shape ", features.shape)
                pickle.dump(features, open('PKU_poseattnet_feat_test10frames.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)


    return PoseAttNet()
