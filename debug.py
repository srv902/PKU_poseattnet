from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import data.pku_mmd as dataset


def test(config):
    print("Debugging dataset")
    train_generator, validation_generator, test_generator = dataset.build(config)
    train_generator.batch_size = config.batch_size
    validation_generator.batch_size = config.eval_batch_size
    test_generator.batch_size = config.eval_batch_size

    n = len(train_generator) - 1
    for i, j in enumerate(train_generator):
        pass  # print('{}/{}'.format(i, n), len(j))
    print("Done")
