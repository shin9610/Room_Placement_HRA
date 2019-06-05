# coding:utf-8
import copy
import glob
import os
import random
from datetime import datetime

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import label


class RoomPlacement:
    def __init__(self, draw_cv2_freq, draw_movie_freq, folder_name, folder_location):
        # 盤面のサイズ
        self.col = 28
        self.row = 28
        self.size = self.col * self.row

        # 可能な行動パターン
        self.enable_actions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        self.num_actions = len(self.enable_actions)

        # 報酬と終了条件の初期化
        self.reward = 0
        self.reward_scheme = {'connect': +1.0, 'shape': +1.0}
        # self.reward_scheme = {'connect': +1.0}
        self.reward_len = len(self.reward_scheme)
        self.reward_channels = []
        self.step_id = 0
        self.game_length = 1500
        self.game_over = False

        # 室の初期設定
        self.random_flag = False
        self.n_agents = 8
        self.room_col = 3
        self.room_row = 3
        self.room_size = self.room_col * self.room_row

        # 室の報酬条件
        # self.your_agent = [1, 0, 3, 2]
        self.your_agent = [1, 0, 3, 2, 5, 4, 7, 6]
        self.neighbor_list = []

        # 室の制約条件
        self.room_upper = 30
        self.room_downer = 10

        # 環境の初期設定　空地 = -1, 外形 = -2, 室 = 0,1,2 ~
        self.square_0 = self.init_square()
        self.site_0 = self.init_site()
        self.state_0 = self.init_state(self.random_flag)
        self.site_0_list = np.array(np.where(self.state_0 == -2)).T.tolist()

        # 更新される環境
        self.state_t = copy.deepcopy(self.state_0)
        self.state_t_1 = 0

        self.state_4chan_0 = self.state_4chan(0, next_flag=False)
        self.next_state_4chan_0 = self.state_4chan(0, next_flag=True)
        self.state_shape = [4, 28, 28]
        # self.state_4chan_t = copy.deepcopy(self.state_4chan_0)

        # 描画系の変数

        self.draw_cv2_freq = draw_cv2_freq
        self.draw_movie_freq = draw_movie_freq

    # 盤面の初期化
    def init_square(self):
        init_square = np.full((self.col, self.row), -1)

        return init_square

    # 敷地位置の初期化
    def init_site(self):
        init_site = copy.deepcopy(self.square_0)

        # # 自動入力の場合
        # for i in range(self.col):
        #     # 上端と下端を外形とする
        #     if i == 0 or i == self.col-1:
        #         init_site[i][:] = -2
        #     # 左右端を外形とする
        #     else:
        #         init_site[i][0] = -2
        #         init_site[i][self.row-1] = -2
        
        # 手入力の場合
        init_site = np.array(
            [[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
             [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
             [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]])

        return init_site

    # 室エージェント位置の初期化
    def init_state(self, flag):
        none_space_arr = np.array(np.where(self.site_0 == -1)).T
        none_space_list = none_space_arr.tolist()

        init_state = copy.deepcopy(self.site_0)

        #ランダム初期位置
        if flag == True:
            for i in range(self.n_agents):
                agent_index = random.choice(none_space_list)
                init_state[agent_index[0]][agent_index[1]] = i
                none_space_list.remove(agent_index)

        # 指定初期位置
        elif self.n_agents == 8:
            init_state = np.array(
            [[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
             [-2, -2, -2, -2, -2, -2, 4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -2, -2, -2, -2, -2, 4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -2, -2, -2, -2, -2, 4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 6, 6, 6, 6, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 6, 6, 6, 6, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 3, 3, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 3, 3, 3, 3, -1, -1, 7, 7, 7, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 3, 3, 3, 3, -1, -1, 7, 7, 7, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, 7, 7, 7, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, 5, 5, 5, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, 5, 5, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, 5, 5, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, 5, 5, 5, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
             [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]])

        elif self.n_agents == 4:
            init_state = np.array(
            [[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
             [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 3, 3, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 3, 3, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, 3, 3, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
             [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
             [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]])

        return init_state

    # stateのチャネルリストの初期化
    def state_4chan(self, now_agent, next_flag):
        temp_my = np.zeros((self.col, self.row))
        temp_your = np.zeros((self.col, self.row))
        temp_other = np.zeros((self.col, self.row))
        temp_site = np.zeros((self.col, self.row))

        # 次のエージェントの分
        if next_flag:
            if now_agent == self.n_agents - 1:
                my_agent_num = 0
            else:
                my_agent_num = now_agent + 1
            your_agent_num = self.your_agent[my_agent_num]

        # 現エージェントの分
        else:
            my_agent_num = now_agent
            your_agent_num = self.your_agent[now_agent]

        # それぞれのエージェントのインデックスを保持する
        temp_my_list = []
        temp_your_list = []
        temp_other_list = []

        # 敷地のインデックスを保持する
        temp_site_arr = np.array(np.where(self.state_0 == -2)).T
        temp_site_list = temp_site_arr.tolist()

        for i in range(self.n_agents):
            if i == my_agent_num:
                temp_my_arr = np.array(np.where(self.state_t == i)).T
                temp_my_list = temp_my_arr.tolist()

            elif i == your_agent_num:
                temp_your_arr = np.array(np.where(self.state_t == i)).T
                temp_your_list = temp_your_arr.tolist()

            else:
                temp_other_arr = np.array(np.where(self.state_t == i)).T
                temp_other_list.extend(temp_other_arr.tolist())

        # tempのチャンネルに数値を入れる
        for i, list in enumerate(temp_my_list):
            temp_my[list[0], list[1]] = 1

        for i, list in enumerate(temp_your_list):
            temp_your[list[0], list[1]] = 1

        for i, list in enumerate(temp_other_list):
            temp_other[list[0], list[1]] = 1

        for i, list in enumerate(temp_site_list):
            temp_site[list[0], list[1]] = 1

        state_4chan = tuple((temp_my, temp_your, temp_other, temp_site))
        # state_4chan = np.array((temp_my, temp_your, temp_other, temp_site))

        return state_4chan

    def step(self, action, now_agent):
        # stepが終了ならば
        if self.step_id >= self.game_length - 1:
            self.game_over = True

            # head_reward = np.zeros(len(self.reward_scheme), dtype=np.float32)
            # # new_obs, reward, game_over, infoを返す
            # return self.state_4chan_t, 0. , self.game_over, {'head_reward': head_reward}

        # step途中の処理
        if self.step_id == 0:
            self.state_t = copy.deepcopy(self.state_0)
        else:
            self.state_t = self.state_t_1

        # state_tの更新
        if 0 <= action <= 3:
            self.update_move(action, now_agent)
        elif 4 <= action <= 7:
            self.update_expand(action, now_agent)
        elif 8 <= action <= 11:
            self.update_reduction(action, now_agent)

        self.state_t_1 = self.state_t

        # 現エージェントと次エージェントのstate_4chanの更新
        self.state_4chan_t = self.state_4chan(now_agent, next_flag=False)
        self.next_state_4chan_t = self.state_4chan(now_agent, next_flag=True)

        # 近傍判定
        self.neighbor_list = self.neighbor_search(now_agent)

        # アスペクト比判定
        self.aspect = self.aspect_search(now_agent)

        # 報酬判定 reward: 報酬の合計，channels: 報酬のチャネル
        # self.reward, self.reward_channels  = self.reward_condition(now_agent)
        self.reward, self.reward_channels = self.reward_condition(now_agent)

        self.step_id += 1

    def update_move(self, action, now_agent):
        temp = np.zeros((self.col, self.row))

        # 移動先にずらしたエージェントの位置
        if action == 0:
            next_agent_arr = np.array(np.where(self.state_t == now_agent)).T + [-1, 0]
        elif action == 1:
            next_agent_arr = np.array(np.where(self.state_t == now_agent)).T + [0, 1]
        elif action == 2:
            next_agent_arr = np.array(np.where(self.state_t == now_agent)).T + [1, 0]
        elif action == 3:
            next_agent_arr = np.array(np.where(self.state_t == now_agent)).T + [0, -1]

        next_agent_list = next_agent_arr.tolist()

        # 移動先にずらした箇所を1で塗りつぶす
        for list in next_agent_list:
            temp[list[0], list[1]] = 1

        # 敷地の箇所を2で塗りつぶす
        for list in self.site_0_list:
            temp[list[0], list[1]] = 2

        # エージェントの位置を3で塗りつぶす
        for i in range(self.n_agents):
            agent_arr = np.array(np.where(self.state_t == i)).T
            agent_list = agent_arr.tolist()
            for list in agent_list:
                temp[list[0], list[1]] = 3

        # 移動時の追加分エージェント
        index_add = np.array(np.where(temp == 1)).T.tolist()

        temp_ = np.zeros((self.col, self.row))

        # now_agentを1で塗りつぶす
        now_agent_list = np.array(np.where(self.state_t == now_agent)).T.tolist()
        for list in now_agent_list:
            temp_[list[0], list[1]] = 1

        # 移動先にずらした箇所を3で塗りつぶす
        for list in next_agent_list:
            temp_[list[0], list[1]] = 3

        # 移動時の削除分エージェント
        index_del = np.array(np.where(temp_ == 1)).T.tolist()

        # now_agent_listの更新
        now_agent_list.extend(index_add)
        for list in index_del:
            now_agent_list.remove(list)

        # 分裂判定→　移動後のエージェント更新
        if self.split_search(now_agent_list):
            for list in index_add:
                self.state_t[list[0], list[1]] = now_agent
            for list in index_del:
                self.state_t[list[0], list[1]] = -1
        else:
            pass

    def update_expand(self, action, now_agent):
        temp = np.zeros((self.col, self.row))
        now_agent_list = np.array(np.where(self.state_t == now_agent)).T.tolist()

        # 移動先にずらしたエージェントの位置
        if action == 4:
            next_agent_arr = np.array(np.where(self.state_t == now_agent)).T + [-1, 0]
        elif action == 5:
            next_agent_arr = np.array(np.where(self.state_t == now_agent)).T + [0, 1]
        elif action == 6:
            next_agent_arr = np.array(np.where(self.state_t == now_agent)).T + [1, 0]
        elif action == 7:
            next_agent_arr = np.array(np.where(self.state_t == now_agent)).T + [0, -1]

        next_agent_list = next_agent_arr.tolist()

        # 移動先にずらした箇所を1で塗りつぶす
        for list in next_agent_list:
            temp[list[0], list[1]] = 1

        # 敷地の箇所を2で塗りつぶす
        for list in self.site_0_list:
            temp[list[0], list[1]] = 2

        # エージェントの位置を3で塗りつぶす
        for i in range(self.n_agents):
            agent_arr = np.array(np.where(self.state_t == i)).T
            agent_list = agent_arr.tolist()
            for list in agent_list:
                temp[list[0], list[1]] = 3

        # 移動時の追加分エージェント
        index_add = np.array(np.where(temp == 1)).T.tolist()

        # 面積の制約条件で分岐
        if len(index_add) != 0:
            if len(now_agent_list) + len(index_add) <= self.room_upper:
                # 拡張後のエージェント更新
                for list in index_add:
                    self.state_t[list[0], list[1]] = now_agent

    def update_reduction(self, action, now_agent):
        temp = np.zeros((self.col, self.row))
        now_agent_list = np.array(np.where(self.state_t == now_agent)).T.tolist()

        # up側が縮小なら
        if action == 8:
            # downにnextを作る
            next_agent_arr = np.array(np.where(self.state_t == now_agent)).T + [1, 0]
        elif action == 9:
            next_agent_arr = np.array(np.where(self.state_t == now_agent)).T + [0, -1]
        elif action == 10:
            next_agent_arr = np.array(np.where(self.state_t == now_agent)).T + [-1, 0]
        elif action == 11:
            next_agent_arr = np.array(np.where(self.state_t == now_agent)).T + [0, 1]

        next_agent_list = next_agent_arr.tolist()

        # 元の位置を1で塗りつぶす
        now_agent_list = np.array(np.where(self.state_t == now_agent)).T.tolist()
        for list in now_agent_list:
            temp[list[0], list[1]] = 1

        # ずらした箇所を3で塗りつぶす
        for list in next_agent_list:
            temp[list[0], list[1]] = 3

        # 削除分エージェント
        index_del = np.array(np.where(temp == 1)).T.tolist()

        # now_agent_listの更新
        for list in index_del:
            now_agent_list.remove(list)

        # 分裂判定
        if self.split_search(now_agent_list):
            # 削除分の個数が元のエージェントの個数を下回るとき
            if len(now_agent_list) > len(index_del):
                # 面積の制約条件を満たすとき
                if len(now_agent_list) - len(index_del) >= self.room_downer:
                    # 削除後の更新
                    for list in index_del:
                        self.state_t[list[0], list[1]] = -1

    def split_search(self, now_agent_list):
        temp = np.zeros((self.col, self.row))

        # 更新後のnow_agentを1で塗りつぶす
        for list in now_agent_list:
            temp[list[0], list[1]] = 1

        # 分裂判定
        _, num_features = label(temp)
        if num_features == 1:
            return True
        else:
            return False

    def neighbor_search(self, now_agent):
        temp = np.zeros((self.col, self.row))
        now_agent_list = np.array(np.where(self.state_t == now_agent)).T.tolist()

        # 元の位置を1で塗りつぶす
        for list in now_agent_list:
            temp[list[0], list[1]] = 1

        # 近傍を1で埋める
        temp = ndimage.binary_dilation(temp).astype(temp.dtype)

        # 元の位置を0で塗りつぶす
        for list in now_agent_list:
            temp[list[0], list[1]] = 0

        # 近傍のインデックスを調べる
        neighbor_index = np.array(np.where(temp == 1)).T.tolist()

        # 近傍のエージェントを調べる
        neighbor_list = []
        for list in neighbor_index:
            neighbor_list.append(int(self.state_t[list[0], list[1]]))

        return neighbor_list

    def aspect_search(self, now_agent):
        """
        # Arguments
            now_agent: 現在のエージェントの番号　→　stateから

        # returns
            アスペクト比と充填率を考慮した数値を出力。1に近いほど正方形に近づく
        """

        # 現在のエージェントの配列を取得
        now_agent_arr = np.array(np.where(self.state_t == now_agent)).T


        # エージェントリストからバウンディングのy, x軸方向のmax, minを取得
        max = now_agent_arr.max(axis=0)
        min = now_agent_arr.min(axis=0)

        # エージェントのバウンディングのy, x長さを取得
        y_distance = max[0]-min[0] + 1
        x_distance = max[1]-min[1] + 1

        # アスペクト比を取得
        if y_distance >= x_distance:
            aspect = x_distance/y_distance
        else:
            aspect = y_distance/x_distance

        # 補正値を取得(室の面積 / バウンディングの面積)
        fill_max = y_distance*x_distance
        fill_agent = len(now_agent_arr)

        modified_value = (fill_agent / fill_max) ** 2

        # アスペクト比に充填の補正値を掛けた数値を出力
        modified_aspect = aspect * modified_value

        return modified_aspect

    def reward_condition(self, now_agent):
        # チャンネル作成
        head_reward = np.zeros(len(self.reward_scheme), dtype=np.float32)

        # 接続報酬を判定
        if self.your_agent[now_agent] in self.neighbor_list:
            reward_connect = self.reward_scheme['connect']
            head_reward[0] = self.reward_scheme['connect']

            # アスペクト比報酬を判定
            if self.aspect >= 0.8:
                reward_shape = self.reward_scheme['shape']
                head_reward[1] = self.reward_scheme['shape']
            else:
                reward_shape = 0.0

        else:
            reward_connect = 0.0
            reward_shape = 0.0

        return reward_connect + reward_shape, head_reward
        # return reward_connect, head_reward

    def number_to_color(self, num):
        # 赤系統
        if num == 0:
            bgr = (18, 0, 230)
        elif num == 1:
            bgr = (79, 0, 229)

        # 黄系統
        elif num == 2:
            bgr = (0, 241, 255)
        elif num == 3:
            bgr = (0, 208, 223)

        # 緑系統
        elif num == 4:
            bgr = (68, 153, 0)
        elif num == 5:
            bgr = (31, 195, 143)

        # 青系統
        elif num == 6:
            bgr = (233, 160, 0)
        elif num == 7:
            bgr = (183, 104, 0)

        else:
            bgr = (100, 100, 100)

        return bgr

    def draw_cv2(self, now_agent, action, step, reward, reward_total, episode, folder_name_images):
        if episode % self.draw_cv2_freq == 0 and episode != 0:
            self.img = np.full((400, 280, 3), 0, dtype=np.uint8)
            grid_pitch = 10
            action_names = ('move_up', 'move_right', 'move_down', 'move_left',
                            'expand_up', 'expand_right', 'expand_down', 'expand_left',
                            'reduction_up', 'reduction_right', 'reduction_down', 'reduction_left')

            # siteの出力
            site_list = self.site_0_list
            for i, list in enumerate(site_list):
                cv2.rectangle(self.img,
                              (int(list[:][1] * grid_pitch + grid_pitch),
                               int(list[:][0] * grid_pitch)),
                              (int(list[:][1] * grid_pitch),
                               int(list[:][0] * grid_pitch + grid_pitch)),
                              (77, 77, 77),
                              thickness=-1)
                cv2.rectangle(self.img,
                              (int(list[:][1] * grid_pitch + grid_pitch),
                               int(list[:][0] * grid_pitch)),
                              (int(list[:][1] * grid_pitch),
                               int(list[:][0] * grid_pitch + grid_pitch)),
                              (200, 200, 200))

            # agentのindex作成
            agent_indexes = []
            for i in range(self.n_agents):
                agent_index = np.array(np.where(self.state_t == i)).T.tolist()
                agent_indexes.append(agent_index)

            # agentの出力
            for i, list in enumerate(agent_indexes):
                if len(list) == 1:
                    cv2.rectangle(self.img,
                                  (int(list[0][1] * grid_pitch + grid_pitch),
                                   int(list[0][0] * grid_pitch)),
                                  (int(list[0][1] * grid_pitch),
                                   int(list[0][0] * grid_pitch + grid_pitch)),
                                  self.number_to_color(i),
                                  thickness=-1)
                    cv2.rectangle(self.img,
                                  (int(list[0][1] * grid_pitch + grid_pitch),
                                   int(list[0][0] * grid_pitch)),
                                  (int(list[0][1] * grid_pitch),
                                   int(list[0][0] * grid_pitch + grid_pitch)),
                                  (200, 200, 200))
                else:
                    for j, j_list in enumerate(list):
                        cv2.rectangle(self.img,
                                      (int(j_list[:][1] * grid_pitch + grid_pitch),
                                       int(j_list[:][0] * grid_pitch)),
                                      (int(j_list[:][1] * grid_pitch),
                                       int(j_list[:][0] * grid_pitch + grid_pitch)),
                                      self.number_to_color(i),
                                      thickness=-1)
                        cv2.rectangle(self.img,
                                      (int(j_list[:][1] * grid_pitch + grid_pitch),
                                       int(j_list[:][0] * grid_pitch)),
                                      (int(j_list[:][1] * grid_pitch),
                                       int(j_list[:][0] * grid_pitch + grid_pitch)),
                                      (200, 200, 200))

            # reward textの出力
            start_x = 5
            start_y = 300
            font = cv2.FONT_HERSHEY_PLAIN
            font_size = 1
            color = (255, 255, 255)
            action_name = action_names[action]
            cv2.putText(self.img, 'agent: ' + str(now_agent), (start_x, start_y), font, font_size, color)
            cv2.putText(self.img, 'action: ' + str(action_name), (start_x, start_y + 15), font, font_size, color)
            cv2.putText(self.img, 'step: ' + str(step), (start_x, start_y + 30), font, font_size, color)
            cv2.putText(self.img, 'reward: ' + str(reward), (start_x, start_y + 45), font, font_size, color)
            cv2.putText(self.img, 'reward_total: ' + str(reward_total), (start_x, start_y + 60), font, font_size, color)

            # fileへの出力
            if len(str(step)) == 4:
                cv2.imwrite(str(folder_name_images) + '/img' + str(step) + '.png', self.img)
                # cv2.imwrite('./cv2_image/images/img' + str(step) + '.png', self.img)
            elif len(str(step)) == 3:
                cv2.imwrite(str(folder_name_images) + '/img0' + str(step) + '.png', self.img)
                # cv2.imwrite('./cv2_image/images/img0' + str(step) + '.png', self.img)
            elif len(str(step)) == 2:
                cv2.imwrite(str(folder_name_images) + '/img00' + str(step) + '.png', self.img)
                # cv2.imwrite('./cv2_image/images/img00' + str(step) + '.png', self.img)
            elif len(str(step)) == 1:
                cv2.imwrite(str(folder_name_images) + '/img000' + str(step) + '.png', self.img)
                # cv2.imwrite('./cv2_image/images/img000' + str(step) + '.png', self.img)

    def gif_animation(self, episode):
        if episode % self.draw_movie_freq == 0 and episode != 0:
            folderName = "./cv2_image/images"

            # 画像ファイルの一覧を取得
            picList = glob.glob(folderName + "\*.png")

            # figオブジェクトを作る
            fig = plt.figure()

            # 空のリストを作る
            ims = []

            # 画像ファイルを順々に読み込んでいく
            for i in range(len(picList)):
                # 1枚1枚のグラフを描き、appendしていく
                tmp = Image.open(picList[i])
                tmp = np.asarray(tmp)
                ims.append([plt.imshow(tmp, animated=True)])

                # アニメーション作成
            ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True)
            # ani.save("./cv2_image/gif_ani/test.gif")
            dt = datetime.now().strftime("%m%d_%H%M")
            # ani.save('./cv2_image/1022_0043/animation' +str(dt) + '.gif', writer="imagemagick")
            ani.save('./cv2_image/gif_ani/animation' +str(dt) + '.gif', writer="imagemagick")

    def movie(self, episode, folder_name_images, folder_name_movies):
        if episode % self.draw_movie_freq == 0 and episode != 0:
            # VideoCapture を作成する。
            img_path = os.path.join(folder_name_images, 'img%04d.png')  # 画像ファイルのパス
            cap = cv2.VideoCapture(img_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 90
            print('write movie')
            dt = datetime.now().strftime("%m%d_%H%M")

            # VideoWriter を作成する。
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            writer = cv2.VideoWriter(str(folder_name_movies) + '/movie' + str(dt) + '.mp4', fourcc, fps, (width, height))

            while True:
                # 1フレームずつ取得する。
                ret, frame = cap.read()
                if not ret:
                    break  # 映像取得に失敗

                writer.write(frame)  # フレームを書き込む。

            writer.release()
            cap.release()

    def reward(self):
        pass

    def observe(self):
        # 現エージェントの次の状態，次エージェントの次の状態を観測して返す
        return self.state_4chan_t, self.next_state_4chan_t, self.reward, self.reward_channels, self.game_over

    def execute_action(self, action, now_agent):
        self.step(action, now_agent)

    def reset(self):
        self.reward = 0
        self.reward_channels = np.zeros(len(self.reward_scheme), dtype=np.float32)
        self.game_over = False
        self.step_id = 0

        # 現エージェントの状態の初期化
        # self.state_t = self.state_0
        self.state_t = self.init_state(flag=True)
        self.state_4chan_t = self.state_4chan_0

        # 次エージェントの状態の初期化
        self.next_state_4chan_t = self.next_state_4chan_0

