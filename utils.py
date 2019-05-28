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
from collections import deque

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
        params['remove_features'] = False # 元のコードはTrue
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

class ExperienceReplay(object):
    def __init__(self, max_size=100, history_len=1, state_shape=None, action_dim=1, reward_dim=1, state_dtype=np.uint8,
                 rng=None):
        if rng is None:
            # random number generator
            self.rng = np.random.RandomState(1234)
        else:
            self.rng = rng
        self.size = 0
        self.head = 0
        self.tail = 0
        self.size = 0
        self.max_size = max_size
        self.history_len = history_len
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.reward_dim = reward_dim  # 10で入っている？
        self.state_dtype = state_dtype
        self._minibatch_size = None

        # ここでreplay memory を保持
        self.D = deque(maxlen=self.max_size)
        self.temp_D = deque(maxlen=self.max_size)
        self.temp_rot_D = deque(maxlen=self.max_size)

        self.states = np.zeros([self.max_size] + list(self.state_shape), dtype=self.state_dtype)
        self.terms = np.zeros(self.max_size, dtype='bool')

        self.actions = np.zeros(self.max_size, dtype='int32')

        if self.reward_dim == 1:
            self.rewards = np.zeros(self.max_size, dtype='float32')
        else:
            self.rewards = np.zeros((self.max_size, self.reward_dim), dtype='float32')

    def compute_ave(self, score, temp_score, ave_score, episode, div=20):
        print("score: " + str(score))
        if episode % div == 0 and episode != 0:
            temp_ave_score = temp_score / div
            if temp_ave_score > ave_score:
                ave_score = temp_ave_score
                print("average_score: " + str(ave_score))
            temp_score = 0

        return ave_score, temp_score

    # 元コードのやつ．storeで置き換え
    def add(self, s, a, r, t):
        self.states[self.tail] = s
        self.actions[self.tail] = a
        self.rewards[self.tail] = r
        self.terms[self.tail] = t
        self.tail = (self.tail + 1) % self.max_size
        if self.size == self.max_size:
            self.head = (self.head + 1) % self.max_size
        else:
            self.size += 1

    def store_temp_exp(self, states, action, reward, states_1, terminal):
        self.temp_D.append((states, action, reward, states_1, terminal))
        # print(np.array(states, action, reward, states_1, terminal))
        # self.temp_D.append(np.array(states, action, reward, states_1, terminal))

    def store_exp(self, score, ave_score):
        if score > ave_score:

            # temp_Dのデータを回転してtemp_Dに格納する
            # self.rotate_experience()

            # temp_DをDに格納する。
            self.D.extend(self.temp_D)
            # print('store, exp_num: ' + str(len(self.D)))

        # temp_Dの消去
        self.temp_D.clear()
        self.temp_rot_D.clear()

        # return (len(self.D) >= self.replay_memory_size)
        # return (len(self.D) >= self.replay_start)

    def rotate_experience(self):
        # temp_Dの分だけ回転データを作成
        for i in range(len(self.temp_D)):

            # 90, 180, 270度回転させたデータを作成
            for key in range(1, 4):

                # state_t作成 チャネルごとに作成して、4*28*28のタプル作成 一旦リスト→　タプルに変換
                state_t_rot = []

                for channel in range(len(self.temp_D[i][0])):
                    state_t_rot_channel = np.rot90(self.temp_D[i][0][channel], k=key)
                    state_t_rot.append(state_t_rot_channel)

                state_t_rot = tuple(state_t_rot)

                # action
                # 90
                if key == 1:
                    action_rot = 3 if self.temp_D[i][1] == 0 \
                        else 0 if self.temp_D[i][1] == 1 \
                        else 1 if self.temp_D[i][1] == 2 \
                        else 2 if self.temp_D[i][1] == 3 \
                        else 7 if self.temp_D[i][1] == 4 \
                        else 4 if self.temp_D[i][1] == 5 \
                        else 5 if self.temp_D[i][1] == 6 \
                        else 6 if self.temp_D[i][1] == 7 \
                        else 11 if self.temp_D[i][1] == 8 \
                        else 8 if self.temp_D[i][1] == 9 \
                        else 9 if self.temp_D[i][1] == 10 \
                        else 10
                # 180
                elif key == 2:
                    action_rot = 2 if self.temp_D[i][1] == 0 \
                        else 3 if self.temp_D[i][1] == 1 \
                        else 0 if self.temp_D[i][1] == 2 \
                        else 1 if self.temp_D[i][1] == 3 \
                        else 6 if self.temp_D[i][1] == 4 \
                        else 7 if self.temp_D[i][1] == 5 \
                        else 4 if self.temp_D[i][1] == 6 \
                        else 5 if self.temp_D[i][1] == 7 \
                        else 10 if self.temp_D[i][1] == 8 \
                        else 11 if self.temp_D[i][1] == 9 \
                        else 8 if self.temp_D[i][1] == 10 \
                        else 9

                    # 270
                # 270
                elif key == 3:
                    action_rot = 2 if self.temp_D[i][1] == 0 \
                        else 3 if self.temp_D[i][1] == 1 \
                        else 0 if self.temp_D[i][1] == 2 \
                        else 1 if self.temp_D[i][1] == 3 \
                        else 5 if self.temp_D[i][1] == 4 \
                        else 6 if self.temp_D[i][1] == 5 \
                        else 7 if self.temp_D[i][1] == 6 \
                        else 4 if self.temp_D[i][1] == 7 \
                        else 9 if self.temp_D[i][1] == 8 \
                        else 10 if self.temp_D[i][1] == 9 \
                        else 11 if self.temp_D[i][1] == 10 \
                        else 8

                # state_t_1 上と同じく
                state_t_1_rot = []

                for channel in range(len(self.temp_D[i][3])):
                    state_t_1_rot_channel = np.rot90(self.temp_D[i][3][channel], k=key)
                    state_t_1_rot.append(state_t_1_rot_channel)

                state_t_1_rot = tuple(state_t_1_rot)

                # temp_rot_Dに追加
                self.temp_rot_D.append((state_t_rot, action_rot, self.temp_D[i][2], state_t_1_rot, self.temp_D[i][4]))

        self.temp_D.extend(self.temp_rot_D)


