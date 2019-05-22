import os
import logging

import numpy as np
from datetime import datetime
from numpy import all, uint8
import pandas as pd
import matplotlib as mpl
from keras import backend as K

mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import model_from_config


def flatten(l):
    return [item for sublist in l for item in sublist]


def clone_model(model, custom_objects={}):
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone


def set_params(params, mode, gamma=None, lr=None, folder_name=None):
    if mode == 'dqn':
        params['gamma'] = .85
        params['learning_rate'] = .0005
        params['remove_features'] = False
        params['use_mean'] = False
        params['use_hra'] = False
    elif mode == 'dqn+1':
        params['gamma'] = .85
        params['learning_rate'] = .0005
        params['remove_features'] = True
        params['use_mean'] = False
        params['use_hra'] = False
    elif mode == 'hra':
        params['gamma'] = .99
        params['learning_rate'] = .001
        params['remove_features'] = False
        params['use_mean'] = True
        params['use_hra'] = True
    elif mode == 'hra+1':
        params['gamma'] = .99
        params['learning_rate'] = .001
        params['remove_features'] = True
        params['use_mean'] = True
        params['use_hra'] = True
    if gamma is not None:
        params['gamma'] = gamma
        params['learning_rate'] = lr
    if folder_name is None:
        params['folder_name'] = mode + '__g' + str(params['gamma']) + '__lr' + str(params['learning_rate']) + '__'
    else:
        params['folder_name'] = folder_name
    return params

def slice_tensor_tensor(tensor, tensor_slice):
    if K.backend() == 'tensorflow':
        amask = K.tf.one_hot(tensor_slice, tensor.get_shape()[1], 1.0, 0.0)
        output = K.tf.reduce_sum(tensor * amask, axis=1)
    else:
        raise Exception("Not using tensor flow as backend")
    return output

def plot_and_write(plot_dict, loc, x_label="", y_label="", title="", kind='line', legend=True,
                   moving_average=False):
    for key in plot_dict:
        plot(data={key: plot_dict[key]}, loc=loc + ".pdf", x_label=x_label, y_label=y_label, title=title,
             kind=kind, legend=legend, index_col=None, moving_average=moving_average)
        write_to_csv(data={key: plot_dict[key]}, loc=loc + ".csv")

def create_folder(folder_location, folder_name):
    i = 0
    t = datetime.now().strftime("%m%d_%H%M")
    while os.path.exists(os.getcwd() + folder_location + folder_name + str(i)):
        i += 1
    # folder_name = os.getcwd() + folder_location + folder_name + str(i)
    folder_location = folder_location.replace('/', '')
    folder_name = os.path.join(os.path.abspath('../'), folder_location, folder_name + str(i) + str(t))
    os.mkdir(folder_name)
    return folder_name

class Font:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bgblue = '\033[44m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'