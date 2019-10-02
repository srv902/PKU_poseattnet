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
            print('train', train_generator, len(train_generator.samples))
            print('validation', validation_generator, len(validation_generator.samples))
            print('test', test_generator, len(test_generator.samples))

        def run(self):
            model = contrib.lstm.model_scripts.models.build_model_without_TS(config.num_neurons,
                                                                             config.dropout,
                                                                             config.batch_size,
                                                                             config.time_steps,
                                                                             config.data_size,
                                                                             dataset.NUM_CLASSES)
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(lr=0.005, clipnorm=1),
                          metrics=['accuracy'])

            model_checkpoint = contrib.utils.CustomModelCheckpoint(model, os.path.join(weights_dir, 'epoch'))
            csv_logger = CSVLogger(os.path.join(experiment_dir, config.experiment_name + '.csv'))

            model.fit_generator(
                generator=self.train_generator,
                validation_data=self.validation_generator,
                epochs=config.num_epochs,
                callbacks=[csv_logger, model_checkpoint],
                max_queue_size=48,
                workers=multiprocessing.cpu_count() - 2,
                use_multiprocessing=True,
            )

            model_json = model.to_json()
            with open(os.path.join(os.path.join(experiment_dir, config.experiment_name + '.json')), "w") as json_file:
                json_file.write(model_json)
            print("Saved model to disk")
            print(model.evaluate_generator(generator=self.test_generator))

    return LSTMExperiment()
