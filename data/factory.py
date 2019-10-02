from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import data.dahlia
import data.pku_mmd

datasets = {
    'dahlia': data.dahlia,
    'pku-mmd': data.pku_mmd
}


def get_dataset(name):
    return datasets[name]
