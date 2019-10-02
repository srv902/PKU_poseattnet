from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing
import os
import keras
import keras.optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
import contrib.i3d.i3d_train
import contrib.utils
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def build(config, dataset, CW):
    # experiment_dir = os.path.join(config.log_dir, config.experiment_name)
    # weights_dir = os.path.join(experiment_dir, 'weights_' + config.experiment_name)

    class I3DExperiment:
        def __init__(self):
            print("I3D experiment")
            print("num_classes", dataset.NUM_CLASSES)
            self.class_weight = CW
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
            i3d = contrib.i3d.i3d_train.i3d_modified(weights='rgb_imagenet_and_kinetics')
            model = i3d.i3d_flattened(num_classes=dataset.NUM_CLASSES)
            optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9)

            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
            csv_logger = CSVLogger(os.path.join(config.experiment_dir, config.experiment_name + '.csv'))

            if self.class_weight == "True":
                print("CW")
                # names = [i.strip() for i in open("/data/stars/user/rdai/smarthomes/split_iccv_2019/shuf_train_CS_32Labels.txt").readlines()]
                # names = [os.path.splitext(i)[0] for i in names]
                # y_train = np.array([int(name_to_int(i.split('')[0])) for i in (names)]) - 1
                names = [i.strip() for i in open("../data/pku_mmd_label_map.txt").readlines()]
                label_dict = {}
                label_dict['background'] = 0
                idx = 1
                for i in names:
                    label_dict[i] = idx
                    idx += 1
                y_train = np.array(label_dict.values())
                print("ytrain")
                print(y_train)

                class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

            parallel_model = keras.utils.multi_gpu_model(model, gpus=config.num_gpus)
            parallel_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            # model_checkpoint = contrib.utils.CustomModelCheckpoint(model, os.path.join(config.weights_dir, 'epoch'))
            model_checkpoint = contrib.utils.CustomModelCheckpoint(parallel_model, os.path.join(config.weights_dir, 'epoch'))

            """
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            
            model.fit_generator(
                generator=self.train_generator,
                validation_data=self.validation_generator,
                epochs=config.num_epochs,
                callbacks=[csv_logger, reduce_lr, model_checkpoint],
                max_queue_size=48,
                workers=multiprocessing.cpu_count() - 2,
                use_multiprocessing=True
            )
            """
            if self.class_weight == "True":
                print("CW_fit")
                parallel_model.fit_generator(
                    generator=self.train_generator,
                    validation_data=self.validation_generator,
                    epochs=config.num_epochs,
                    callbacks=[csv_logger, reduce_lr, model_checkpoint],
                    max_queue_size=48,
                    workers=multiprocessing.cpu_count() - 2,
                    use_multiprocessing=True,
                    class_weight=class_weight
                )
            else:
                parallel_model.fit_generator(
                    generator=self.train_generator,
                    validation_data=self.validation_generator,
                    epochs=config.num_epochs,
                    callbacks=[csv_logger, reduce_lr, model_checkpoint],
                    max_queue_size=48,
                    workers=multiprocessing.cpu_count() - 2,
                    use_multiprocessing=True
                )

            # model_json = model.to_json()
            model_json = parallel_model.to_json()
            with open(os.path.join(os.path.join(config.experiment_dir, config.experiment_name + '.json')), "w") as f:
                f.write(model_json)
            print("Saved model to disk")
            # print(model.evaluate_generator(generator=self.test_generator))
            print(parallel_model.evaluate_generator(generator=self.test_generator))


    return I3DExperiment()
