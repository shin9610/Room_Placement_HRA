# coding:utf-8
import copy
import glob
import os
import random
from datetime import datetime

import cv2
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image
from scipy import ndimage
from scipy.ndimage import label


class RoomPlacement:
    def __init__(self, draw_cv2_freq, draw_movie_freq, test_draw_cv2_freq, test_draw_movie_freq, step_max_len,
                 folder_name, folder_location, test):
        # 時間計測
        self.update_time = 0
        self.state_channel_time = 0
        self.reward_condition_time = 0
        self.neighbor_search_time = 0

        # 盤面のサイズ
        self.col = 28
        self.row = 28
        # self.col = 14
        # self.row = 14
        # self.img_col = 600
        # self.img_row = 280


        # 0, 1, 2, 3: 移動，4, 5, 6, 7: 拡大， 8, 9, 10, 11: 縮小，12: 停止
        # 0, 1, 2, 3: 移動，4, 5, 6, 7, 8, 9, 10, 11: 変形，12: 停止
        self.enable_actions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        self.num_actions = len(self.enable_actions)

        # 報酬と終了条件の初期化
        self.reward = 0
        # self.reward_scheme = {'connect': +1.0, 'shape': +1.0, 'area': +1.0}
        self.reward_scheme = {'connect0': +1.0, 'connect1': +1.0, 'connect2': +1.0, 'connect3': +1.0,
                              'shape': +1.0, 'area': +1.0}
        # self.reward_scheme = {'connect0': +1.0, 'connect1': +1.0, 'connect2': +1.0, 'connect3': +1.0}
        # self.reward_scheme = {'connect0': +1.0, 'connect1': +1.0, 'connect2': +1.0, 'connect3': +1.0,
        #                       'area': +1.0}
        # self.reward_scheme = {'connect0': +1.0, 'connect1': +1.0, 'connect2': +1.0, 'connect3': +1.0,
        #                       'area': +1.0, 'Effective_dim': +1.0}

        self.reward_name = [name for i, name in enumerate(self.reward_scheme)]
        self.reward_len = len(self.reward_scheme)
        self.reward_channels = []
        self.step_id = 0
        self.game_length = step_max_len
        self.limit_flag = False # 行動を選択した際，制約に引っかかったときはTrue
        self.game_over = False # 規定ステップを超えてしまうとgame_over
        self.term = False # 規定ステップ内でクリアするとterm

        # 室の初期設定
        if not test:
            self.random_flag = False
        else:
            self.random_flag = True

        self.evely_random_flag = False
        self.n_agents = 4
        self.room_col = 3
        self.room_row = 3
        self.room_size = self.room_col * self.room_row


        # 室の報酬条件
        # self.your_agent = [1, 0, 3, 2]
        self.your_agent_max = 4
        # self.your_agent = [1, 0, 3, 2, 5, 4, 7, 6]
        # self.your_agent = [1, 0, 0, 0, 0, 0, 0, 0]

        # self.your_agent = [[1, None, None, None],
        #                    [0, None, None, None],
        #                    [3, None, None, None],
        #                    [2, None, None, None],
        #                    [5, None, None, None],
        #                    [4, None, None, None],
        #                    [7, None, None, None],
        #                    [6, None, None, None]]

        # self.your_agent = [[1, 2, None, None],
        #                    [0, 3, None, None],
        #                    [3, 5, None, None],
        #                    [2, 7, 1, None],
        #                    [5, None, None, None],
        #                    [4, None, None, None],
        #                    [7, None, None, None],
        #                    [6, None, None, None]]

        # self.your_agent = [[1, None, None, None],
        #                    [0, 2, None, None],
        #                    [1, 3, None, None],
        #                    [2, None, None, None],
        #                    [6, 5, None, None],
        #                    [4, 7, None, None],
        #                    [4, None, None, None],
        #                    [5, None, None, None]]

        self.your_agent = [[1, 3, None, None],
                           [0, 2, None, None],
                           [1, 3, None, None],
                           [0, 2, None, None]]

        # self.your_agent = [[1, None, None, None],
        #                    [0, None, None, None],
        #                    [3, None, None, None],
        #                    [2, None, None, None]]

        self.neighbor_list = []

        # 室の面積条件
        self.room_upper = 20
        self.room_downer = 12
        self.effective_len = 4

        # 環境の初期設定　空地 = -1, 外形 = -2, 室 = 0,1,2 ~
        self.square_0 = self.init_square()
        self.site_0 = self.init_site()

        # 初期位置をランダムに生成する場合
        if self.random_flag:
            self.state_seed_0 = self.init_state_seed()
            self.site_0_list = np.array(np.where(self.state_seed_0 == -2)).T.tolist()
            self.state_0 = self.init_state()

        # 初期位置を固定で生成する場合
        else:
            self.state_0 = self.init_state()
            self.site_0_list = np.array(np.where(self.state_0 == -2)).T.tolist()
            self.state_t = copy.deepcopy(self.state_0)
            self.state_t_1 = 0

        # 更新される環境
        # input_channel
        self.state_shape = [7, self.col, self.row]
        self.state_channel_0 = self.state_channel(0, next_flag=False)
        self.next_state_channel_0 = self.state_channel(0, next_flag=True)

        # 描画系の変数
        if not test:
            self.draw_cv2_freq = draw_cv2_freq
            self.draw_movie_freq = draw_movie_freq
        else:
            self.draw_cv2_freq = test_draw_cv2_freq
            self.draw_movie_freq = test_draw_movie_freq

        self.img_contents_x_span = 250
        self.img_contents_y_span = 250
        self.img_contents_x_num = self.n_agents + 1
        self.img_contents_y_num = self.reward_len + 2
        self.img_row = self.img_contents_x_span * self.img_contents_x_num
        self.img_col = self.img_contents_y_span * self.img_contents_y_num
        self.img_grid_pitch = int(self.img_contents_x_span / self.row)
        self.agg_q_temps = []
        self.merged_q_temps = []
        self.agg_w_temps = []


    # 盤面の初期化
    def init_square(self):
        init_square = np.full((self.col, self.row), -1)

        return init_square

    # 敷地位置の初期化
    def init_site(self):
        init_site = copy.deepcopy(self.square_0)

        # 自動入力の場合
        for i in range(self.col):
            # 上端と下端を外形とする
            if i == 0 or i == self.col-1:
                init_site[i][:] = -2
            # 左右端を外形とする
            else:
                init_site[i][0] = -2
                init_site[i][self.row-1] = -2
        
        # # 手入力の場合
        # init_site = np.array(
        #     [[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        #      [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        #      [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]])

        return init_site

    # 室エージェント位置の初期化
    def init_state_seed(self):
        none_space_arr = np.array(np.where(self.site_0 == -1)).T
        none_space_list = none_space_arr.tolist()

        init_seed_state = copy.deepcopy(self.site_0)

        #ランダム初期位置生成
        for i in range(self.n_agents):
            agent_index = random.choice(none_space_list)
            init_seed_state[agent_index[0]][agent_index[1]] = i
            none_space_list.remove(agent_index)

        # # 指定初期位置
        # elif self.n_agents == 8:
        #     init_state = np.array(
        #     [[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        #      [-2, -2, -2, -2, -2, -2, 4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -2, -2, -2, -2, -2, 4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -2, -2, -2, -2, -2, 4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 6, 6, 6, 6, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 6, 6, 6, 6, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 3, 3, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 3, 3, 3, 3, -1, -1, 7, 7, 7, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 3, 3, 3, 3, -1, -1, 7, 7, 7, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, 7, 7, 7, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, 5, 5, 5, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, 5, 5, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, 5, 5, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, 5, 5, 5, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        #      [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]])
        #
        # elif self.n_agents == 4:
        #     init_state = np.array(
        #     [[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        #      [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 3, 3, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 3, 3, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, 3, 3, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        #      [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        #      [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]])
        return init_seed_state

    def init_state(self):
        # seed_stateから報酬面積の下限まで成長させる　→　state
        if self.random_flag:
            term = False
            local_cnt = 0
            while not term:
                for num in range(self.n_agents):
                    # 拡張(4, 5, 6, 7)しか選択しない
                    action = np.random.choice((4, 5, 6, 7))

                    self.step_seed(action, num, local_cnt)
                    local_cnt += 1
                    if local_cnt > self.n_agents:
                        term = True
                        break

            init_state = self.state_t

        else:
            if self.n_agents == 8:
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

    def state_channel(self, now_agent, next_flag):

        temp_my = np.zeros((self.col, self.row))
        temp_your0 = np.zeros((self.col, self.row))
        temp_your1 = np.zeros((self.col, self.row))
        temp_your2 = np.zeros((self.col, self.row))
        temp_your3 = np.zeros((self.col, self.row))
        temp_other = np.zeros((self.col, self.row))
        temp_site = np.zeros((self.col, self.row))

        # 次のエージェントの分
        if next_flag:
            if now_agent == self.n_agents - 1:
                my_agent_num = 0
            else:
                my_agent_num = now_agent + 1
            your_agents = self.your_agent[my_agent_num]

        # 現エージェントの分
        else:
            my_agent_num = now_agent
            your_agents = self.your_agent[now_agent]

        # それぞれのエージェントのインデックスを保持する
        temp_my_list = []
        temp_your0_list = []
        temp_your1_list = []
        temp_your2_list = []
        temp_your3_list = []
        temp_other_list = []

        # 敷地のインデックスを保持する
        temp_site_arr = np.array(np.where(self.site_0 == -2)).T
        temp_site_list = temp_site_arr.tolist()

        if self.state_shape[0]==4:
            for i in range(self.n_agents):
                if i == my_agent_num:
                    temp_my_arr = np.array(np.where(self.state_t == i)).T
                    temp_my_list = temp_my_arr.tolist()
                elif i in your_agents:
                    temp_your0_arr = np.array(np.where(self.state_t == i)).T
                    temp_your0_list.extend(temp_your0_arr.tolist())
                else:
                    temp_other_arr = np.array(np.where(self.state_t == i)).T
                    temp_other_list.extend(temp_other_arr.tolist())

        elif self.state_shape[0]==7:
            for i in range(self.n_agents):
                if i == my_agent_num:
                    temp_my_arr = np.array(np.where(self.state_t == i)).T
                    temp_my_list = temp_my_arr.tolist()
                elif i == your_agents[0]:
                    temp_your0_arr = np.array(np.where(self.state_t == i)).T
                    temp_your0_list = temp_your0_arr.tolist()
                elif i == your_agents[1]:
                    temp_your1_arr = np.array(np.where(self.state_t == i)).T
                    temp_your1_list = temp_your1_arr.tolist()
                elif i == your_agents[2]:
                    temp_your2_arr = np.array(np.where(self.state_t == i)).T
                    temp_your2_list = temp_your2_arr.tolist()
                elif i == your_agents[3]:
                    temp_your3_arr = np.array(np.where(self.state_t == i)).T
                    temp_your3_list = temp_your3_arr.tolist()
                else:
                    temp_other_arr = np.array(np.where(self.state_t == i)).T
                    temp_other_list.extend(temp_other_arr.tolist())

        # tempのチャンネルに数値を入れる
        if self.state_shape[0] == 4:
            for i, list in enumerate(temp_my_list):
                temp_my[list[0], list[1]] = 1
            for i, list in enumerate(temp_your0_list):
                temp_your0[list[0], list[1]] = 1
            for i, list in enumerate(temp_other_list):
                temp_other[list[0], list[1]] = 1
            for i, list in enumerate(temp_site_list):
                temp_site[list[0], list[1]] = 1

            state_channel = tuple((temp_my, temp_your0, temp_other, temp_site))
            return state_channel

        elif self.state_shape[0] == 7:
            for i, list in enumerate(temp_my_list):
                temp_my[list[0], list[1]] = 1

            if your_agents[0] != None:
                for i, list in enumerate(temp_your0_list):
                    temp_your0[list[0], list[1]] = 1
            if your_agents[1] != None:
                for i, list in enumerate(temp_your1_list):
                    temp_your1[list[0], list[1]] = 1
            if your_agents[2] != None:
                for i, list in enumerate(temp_your2_list):
                    temp_your2[list[0], list[1]] = 1
            if your_agents[3] != None:
                for i, list in enumerate(temp_your3_list):
                    temp_your3[list[0], list[1]] = 1

            for i, list in enumerate(temp_other_list):
                temp_other[list[0], list[1]] = 1

            for i, list in enumerate(temp_site_list):
                temp_site[list[0], list[1]] = 1

            state_channel = tuple((temp_my, temp_your0, temp_your1, temp_your2, temp_your3, temp_other, temp_site))
            return state_channel

    def step_seed(self, action, now_agent, local_cnt):
        if local_cnt == 0:
            # self.state_t = copy.deepcopy(self.state_seed_0)
            self.state_t = np.copy(self.state_seed_0)
        else:
            self.state_t = self.state_t_1

        self.update_expand(action, now_agent)

        self.state_t_1 = self.state_t

        self.reward, self.reward_channels = self.reward_condition(now_agent)

    # 1stepを実行する
    def step(self, action, now_agent):
        # stepが終了ならば
        if self.step_id >= self.game_length - 1:
            self.game_over = True
            # print('time_update: ' +str(self.update_time))
            # print('time_state_channel: ' +str(self.state_channel_time))
            # print('time_reward_condition: ' +str(self.reward_condition_time))
            # print('time_neighbor_search: ' +str(self.neighbor_search_time))

            self.update_time = 0
            self.state_channel_time = 0
            self.reward_condition_time = 0
            self.neighbor_search_time = 0

            # head_reward = np.zeros(len(self.reward_scheme), dtype=np.float32)
            # # new_obs, reward, game_over, infoを返す
            # return self.state_4chan_t, 0. , self.game_over, {'head_reward': head_reward}

        # step途中の処理
        if self.step_id == 0:
            # self.state_t = copy.deepcopy(self.state_0)
            self.state_t = np.copy(self.state_0)
            self.limit_flag = False
        else:
            self.state_t = self.state_t_1
            self.limit_flag = False

        # state_tの更新
        start = time.time()
        if 0 <= action <= 3:
            self.update_move(action, now_agent)
        elif 4 <= action <= 7:
            self.update_expand(action, now_agent)
        elif 8 <= action <= 11:
            self.update_reduction(action, now_agent)
        elif action == 12:
            self.update_non(action)

        self.update_time += round(time.time() - start, 8)

        self.state_t_1 = self.state_t

        # 現エージェントと次エージェントのstate_4chanの更新
        # self.state_4chan_t = self.state_4chan(now_agent, next_flag=False)
        # self.next_state_4chan_t = self.state_4chan(now_agent, next_flag=True)


        self.state_channel_t = self.state_channel(now_agent, next_flag=False)
        self.next_state_channel_t = self.state_channel(now_agent, next_flag=True)

        # 報酬判定 reward: 報酬の合計，channels: 報酬のチャネル
        self.reward, self.reward_channels = self.reward_condition(now_agent)
        # 全体としての報酬を観測
        self.reward_all_condition(now_agent)

        self.step_id += 1

    # 1stepを実行する
    def step_deform(self, action, now_agent):
        # stepが終了ならば
        if self.step_id >= self.game_length - 1:
            self.game_over = True

            # head_reward = np.zeros(len(self.reward_scheme), dtype=np.float32)
            # # new_obs, reward, game_over, infoを返す
            # return self.state_4chan_t, 0. , self.game_over, {'head_reward': head_reward}

        # step途中の処理
        if self.step_id == 0:
            # self.state_t = copy.deepcopy(self.state_0)
            self.state_t = np.copy(self.state_0)

        else:
            self.state_t = self.state_t_1

        # state_tの更新
        if 0 <= action <= 3:
            self.update_move_rect(action, now_agent)
        elif 4 <= action <= 7:
            self.update_expand_rect(action, now_agent)
        elif 8 <= action <= 11:
            self.update_reduction_rect(action, now_agent)
        elif action == 12:
            self.update_non(action)

        self.state_t_1 = self.state_t

        # 現エージェントと次エージェントのstate_4chanの更新
        # self.state_4chan_t = self.state_4chan(now_agent, next_flag=False)
        # self.next_state_4chan_t = self.state_4chan(now_agent, next_flag=True)

        self.state_channel_t = self.state_channel(now_agent, next_flag=False)
        self.next_state_channel_t = self.state_channel(now_agent, next_flag=True)

        # 報酬判定 reward: 報酬の合計，channels: 報酬のチャネル
        self.reward, self.reward_channels = self.reward_condition(now_agent)

        self.step_id += 1

    # 移動(面積一定ではない)
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

        # 自分含め，エージェントの位置を3で塗りつぶす
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

        # 制約面積条件　→　報酬の場合は逃す
        # if len(now_agent_list) - len(index_del) > self.room_downer:

        # # addとdelが等しい
        # if len(index_add) == len(index_del):
        #     # now_agent_listの更新
        #     now_agent_list.extend(index_add)
        #     for list in index_del:
        #         now_agent_list.remove(list)
        # # addとdelが等しくない
        # else:
        #     # index_add分だけindex_delの数を調整
        #     index_del = index_del[:len(index_add)]
        #     # now_agent_listの更新
        #     now_agent_list.extend(index_add)
        #     for list in index_del:
        #         now_agent_list.remove(list)

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
        # 分裂しているのでpass
        else:
            pass

        # 衝突判定　→　移動先に何かある場合に移動
        if len(index_add)==0:
            self.limit_flag = True

        # 制約面積条件に乗らないのでpass
        # else:
        #     pass

    # 矩形移動(面積一定)
    def update_move_rect(self, action, now_agent):
        temp = np.zeros((self.col, self.row))
        now_agent_list = np.array(np.where(self.state_t == now_agent)).T.tolist()
        _, neighbor_index, neighbor_direction = self.neighbor_search(now_agent)

        if action == 0:
            exp_agent_arr = neighbor_direction[0]
            red_agent_arr = neighbor_direction[2] - [1, 0]
        elif action == 1:
            exp_agent_arr = neighbor_direction[1]
            red_agent_arr = neighbor_direction[3] + [0, 1]
        elif action == 2:
            exp_agent_arr = neighbor_direction[2]
            red_agent_arr = neighbor_direction[0] + [1, 0]
        elif action == 3:
            exp_agent_arr = neighbor_direction[3]
            red_agent_arr = neighbor_direction[1] - [0, 1]

        exp_agent_list = exp_agent_arr.tolist()
        red_agent_list = red_agent_arr.tolist()

        # 拡張先の箇所を1で塗りつぶす
        for list in exp_agent_list:
            temp[list[0], list[1]] = 1

        # 敷地の箇所を2で塗りつぶす
        for list in self.site_0_list:
            temp[list[0], list[1]] = 2

        # 自分含め，エージェントの位置を3で塗りつぶす
        for i in range(self.n_agents):
            agent_arr = np.array(np.where(self.state_t == i)).T
            agent_list = agent_arr.tolist()
            for list in agent_list:
                temp[list[0], list[1]] = 3

        # 移動時の追加分エージェント
        index_add = np.array(np.where(temp == 1)).T.tolist()

        # 拡張先に障害物がない場合
        if len(exp_agent_list) == len(index_add):
            index_del = red_agent_list

            # 拡張先の追加
            for list in index_add:
                self.state_t[list[0], list[1]] = now_agent
            # 縮小先の削除
            for list in index_del:
                self.state_t[list[0], list[1]] = -1
            # 制約フラグ
            self.limit_flag = False
        else:
            # 制約フラグ
            self.limit_flag = True

    # 拡大
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

        # 面積の制約条件
        if len(index_add) != 0:
        #     if len(now_agent_list) + len(index_add) <= self.room_upper:
            # 拡張後のエージェント更新
            for list in index_add:
                self.state_t[list[0], list[1]] = now_agent
        else:
            # 衝突判定 →　拡大先に障害物あり．
            self.limit_flag = True

    # 矩形拡大
    def update_expand_rect(self, action, now_agent):
        temp = np.zeros((self.col, self.row))
        _, neighbor_index, neighbor_direction = self.neighbor_search(now_agent)
        exp_agent_arr = np.array([])

        if action == 4: # up
            exp_agent_arr = neighbor_direction[0]
        elif action == 5: # right
            exp_agent_arr = neighbor_direction[1]
        elif action == 6: # down
            exp_agent_arr = neighbor_direction[2]
        elif action == 7: # left
            exp_agent_arr = neighbor_direction[3]

        exp_agent_list = exp_agent_arr.tolist()

        # 拡張先の箇所を1で塗りつぶす exp_agent_list
        for list in exp_agent_list:
            temp[list[0], list[1]] = 1

        # 敷地の箇所を2で塗りつぶす
        for list in self.site_0_list:
            temp[list[0], list[1]] = 2

        # 自分含め，エージェントの位置を3で塗りつぶす
        for i in range(self.n_agents):
            agent_arr = np.array(np.where(self.state_t == i)).T
            agent_list = agent_arr.tolist()
            for list in agent_list:
                temp[list[0], list[1]] = 3

        # 実際に拡張先に残ったエージェント
        index_add = np.array(np.where(temp == 1)).T.tolist()

        # 拡張先に障害物がない場合
        if len(exp_agent_list) == len(index_add):
            for list in index_add:
                self.state_t[list[0], list[1]] = now_agent
            # 制約フラグ
            self.limit_flag = False
        else:
            # 制約フラグ
            self.limit_flag = True

    # 縮小
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
                # if len(now_agent_list) - len(index_del) >= self.room_downer:
                # 削除後の更新
                for list in index_del:
                    self.state_t[list[0], list[1]] = -1

    # 矩形縮小
    def update_reduction_rect(self, action, now_agent):
        temp = np.zeros((self.col, self.row))
        now_agent_list = np.array(np.where(self.state_t == now_agent)).T.tolist()
        _, neighbor_index, neighbor_direction = self.neighbor_search(now_agent)
        red_agent_arr = np.array([])

        if action == 8:
            red_agent_arr = neighbor_direction[0] + [1, 0]
        elif action == 9:
            red_agent_arr = neighbor_direction[1] - [0, 1]
        elif action == 10:
            red_agent_arr = neighbor_direction[2] - [1, 0]
        elif action == 11:
            red_agent_arr = neighbor_direction[3] + [0, 1]

        red_agent_list = red_agent_arr.tolist()

        # 現在のエージェント数と削除するエージェント数の差分が0より大きいなら
        if len(now_agent_list)-len(red_agent_list)>0:
            # 削除する
            for list in red_agent_list:
                self.state_t[list[0], list[1]] = -1
            self.limit_flag = False
        else:
            self.limit_flag = True

    # 変形
    def update_deform(self, action, now_agent):
        temp = np.zeros((self.col, self.row))
        now_agent_list = np.array(np.where(self.state_t == now_agent)).T.tolist()
        _, neighbor_index, neighbor_direction = self.neighbor_search(now_agent)

        if action == 4: # UP_left
            exp_agent_arr = neighbor_direction[0][1:]
            red_agent_arr = neighbor_direction[3] + [0, 1]

        elif action == 5: # UP_right
            exp_agent_arr = neighbor_direction[0][:-1]
            red_agent_arr = neighbor_direction[1] - [0, 1]

        elif action == 6: # RIGHT_up
            exp_agent_arr = neighbor_direction[1][1:]
            red_agent_arr = neighbor_direction[0] + [1, 0]

        elif action == 7: # RIGHT_down
            exp_agent_arr = neighbor_direction[1][:-1]
            red_agent_arr = neighbor_direction[2] - [1, 0]

        elif action == 8: # DOWN_right
            exp_agent_arr = neighbor_direction[2][:-1]
            red_agent_arr = neighbor_direction[1] - [0, 1]

        elif action == 9: # DOWN_left
            exp_agent_arr = neighbor_direction[2][1:]
            red_agent_arr = neighbor_direction[3] + [0, 1]

        elif action == 10: # LEFT_down
            exp_agent_arr = neighbor_direction[3][:-1]
            red_agent_arr = neighbor_direction[2] - [1, 0]

        elif action == 11: # LEFT_up
            exp_agent_arr = neighbor_direction[3][1:]
            red_agent_arr = neighbor_direction[0] + [1, 0]

        exp_agent_list = exp_agent_arr.tolist()
        red_agent_list = red_agent_arr.tolist()

        # 拡張先の箇所を1で塗りつぶす
        for list in exp_agent_list:
            temp[list[0], list[1]] = 1

        # 敷地の箇所を2で塗りつぶす
        for list in self.site_0_list:
            temp[list[0], list[1]] = 2

        # 自分含め，エージェントの位置を3で塗りつぶす
        for i in range(self.n_agents):
            agent_arr = np.array(np.where(self.state_t == i)).T
            agent_list = agent_arr.tolist()
            for list in agent_list:
                temp[list[0], list[1]] = 3

        # 移動時の追加分エージェント
        index_add = np.array(np.where(temp == 1)).T.tolist()

        # 拡張先に障害物がない場合
        if len(exp_agent_list) == len(index_add):
            index_del = red_agent_list
            # 面積条件の判定
            if self.room_downer <= len(now_agent_list) + len(index_add) - len(index_del) <= self.room_upper:
                # 拡張先の追加
                for list in index_add:
                    self.state_t[list[0], list[1]] = now_agent
                # 縮小先の削除
                for list in index_del:
                    self.state_t[list[0], list[1]] = -1

    # 停止
    def update_non(self, action):
        if action == 12:
            pass

    # 分裂判定
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

    # 近傍の探索
    def neighbor_search(self, now_agent):
        temp = np.zeros((self.col, self.row))
        now_agent_list = np.array(np.where(self.state_t == now_agent)).T.tolist()

        # 元の位置を1で塗りつぶす
        for list in now_agent_list:
            temp[list[0], list[1]] = 1

        # 近傍を1で埋める　0.28秒
        temp = ndimage.binary_dilation(temp).astype(temp.dtype)

        # 元の位置を0で塗りつぶす
        for list in now_agent_list:
            temp[list[0], list[1]] = 0

        # 近傍のインデックスを調べる
        neighbor_index = np.array(np.where(temp == 1)).T.tolist()

        # -----
        # 近傍のインデックスを方角ごとに分類　0.23秒　→　

        # 0.17秒
        neighbor_up = np.min(np.array(neighbor_index), axis=0)[0]
        neighbor_right = np.max(np.array(neighbor_index), axis=0)[1]
        neighbor_down = np.max(np.array(neighbor_index), axis=0)[0]
        neighbor_left = np.min(np.array(neighbor_index), axis=0)[1]

        # 0.07秒
        start = time.time()

        neighbor_up_index = np.array([index for index in neighbor_index if index[0]==neighbor_up])
        neighbor_right_index = np.array([index for index in neighbor_index if index[1]==neighbor_right])
        neighbor_down_index = np.array([index for index in neighbor_index if index[0]==neighbor_down])
        neighbor_left_index = np.array([index for index in neighbor_index if index[1]==neighbor_left])

        self.neighbor_search_time += round(time.time() - start, 8)


        # 0.01秒
        neighbor_direction = np.array([neighbor_up_index, neighbor_right_index,
                                       neighbor_down_index, neighbor_left_index])


        # -----
        # 近傍のエージェントを調べる 0.02秒
        neighbor_list = []
        for list in neighbor_index:
            neighbor_list.append(int(self.state_t[list[0], list[1]]))


        return neighbor_list, neighbor_index, neighbor_direction

    # 形状の判定
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

        return modified_aspect, x_distance, y_distance

    # 面積の判定
    def area_search(self, now_agent):
        now_agent_list = np.array(np.where(self.state_t == now_agent)).T
        return len(now_agent_list)

    # 形状判定で出力されるx, y disを有効寸法の判定に用いる
    def effective_len_search(self, now_agent):
        _, x_len, y_len = self.aspect_search(now_agent)
        if x_len >= self.effective_len and y_len >= self.effective_len:
            return True
        else:
            return False

    def reward_condition(self, now_agent):
        # チャンネル作成
        start = time.time()

        head_reward = np.zeros(len(self.reward_scheme), dtype=np.float32)

        # 接続報酬を判定
        # if self.your_agent[now_agent] in self.neighbor_search(now_agent):
        #     head_reward[0] = self.reward_scheme['connect']
        #
        #     # アスペクト比報酬を判定
        #     if self.aspect_search(now_agent) >= 0.8:
        #         head_reward[1] = self.reward_scheme['shape']
        #     else:
        #         pass
        #
        #     # 面積報酬を判定
        #     if self.room_downer <= self.area_search(now_agent) <= self.room_upper:
        #         head_reward[2] = self.reward_scheme['area']
        #     else:
        #         pass
        # else:
        #     pass

        # 接続報酬を判定
        # print(self.your_agent[now_agent])
        # print(self.neighbor_search(now_agent))

        # if self.your_agent[now_agent] in self.neighbor_search(now_agent):
        #     head_reward[0] = self.reward_scheme['connect']

        # この与え方だと接続に対して一律に観測する．　→　headを作成した方が適切？

        neighbors, _, _ = self.neighbor_search(now_agent)

        for n, your_list in enumerate(self.your_agent[now_agent]):
            # 接続に単数ヘッド
            if 'connect' in self.reward_name:
                if your_list in neighbors:
                    head_reward[self.reward_name.index('connect')] += self.reward_scheme['connect']
            # 接続に複数ヘッドあり
            elif 'connect' not in self.reward_name:
                if your_list in neighbors:
                    if not self.limit_flag:
                        head_reward[self.reward_name.index('connect' + str(n))] = self.reward_scheme['connect' + str(n)]
                    else:
                        head_reward[self.reward_name.index('connect' + str(n))] = -0.1
                elif your_list == None:
                    head_reward[self.reward_name.index('connect' + str(n))] = None

        # アスペクト比報酬を判定
        aspect, _, _ = self.aspect_search(now_agent)
        if aspect >= 0.8:
            head_reward[self.reward_name.index('shape')] = self.reward_scheme['shape']

        # 面積報酬を判定
        if self.room_downer <= self.area_search(now_agent) <= self.room_upper:
            head_reward[self.reward_name.index('area')] = self.reward_scheme['area']

        # スタックを判定

        # # 有効寸法を判定
        # if self.effective_len_search(now_agent):
        #     head_reward[5] = self.reward_scheme['Effective_dim']

        self.reward_condition_time += round(time.time() - start, 8)

        # nanを除いた報酬の合計を返す
        return sum([i for i in head_reward if not np.isnan(i)]), head_reward
        # return reward_connect, head_reward

    def reward_all_condition(self, now_agent):
        cnt = 0
        for i in range(self.n_agents):
            if not i == now_agent:
                _, head_reward = self.reward_condition(i)
                if sum(head_reward) == 4.0: # 報酬条件ベタ打ち…
                    cnt += 1

        if cnt == self.n_agents-1:
            self.term = True
            self.game_over = True
            print('all agents get reward')

    def number_to_color(self, num):
        # 赤系統
        if num == 0:
            bgr = (18, 0, 230)
        elif num == 4:
            bgr = (79, 0, 229)

        # 黄系統
        elif num == 1:
            bgr = (0, 241, 255)
        elif num == 5:
            bgr = (0, 208, 223)

        # 緑系統
        elif num == 2:
            bgr = (68, 153, 0)
        elif num == 6:
            bgr = (31, 195, 143)

        # 青系統
        elif num == 3:
            bgr = (233, 160, 0)
        elif num == 7:
            bgr = (183, 104, 0)

        else:
            bgr = (100, 100, 100)

        return bgr

    def draw_cv2(self, now_agent, action, step, reward, agg_w, agg_q, merged_q, reward_total, episode, folder_name_images):
        if episode % self.draw_cv2_freq == 0 and episode != 0:
            self.img = np.full((self.img_col, self.img_row, 3), 0, dtype=np.uint8)
            # # 背景色の変更
            # self.img.fill(255)

            # textの設定
            font = cv2.FONT_HERSHEY_PLAIN
            font_size = 1
            color = (255, 255, 255)
            a_names = ('mo_up', 'mo_right', 'mo_down', 'mo_left',
                            'ex_up', 'ex_right', 'ex_down', 'ex_left',
                            're_up', 're_right', 're_down', 're_left', 'stop')

            if step == 1:
                self.agg_q_temps = [agg_q for i in range(self.n_agents)]
                self.merged_q_temps = [merged_q for i in range(self.n_agents)]
                self.agg_w_temps = [agg_w for i in range(self.n_agents)]
            else:
                self.agg_q_temps[now_agent] = agg_q
                self.merged_q_temps[now_agent] = merged_q
                self.agg_w_temps[now_agent] = agg_w

            # グリッドの出力
            knot_pts = self.draw_grid(n_contents_x=5, n_contents_y=6, font=font, font_size=font_size, color=color)

            # 室配置，Q_headsの出力
            for x in range(self.n_agents+1):
                for y in range(len(agg_q)+2):
                    if x < self.n_agents:
                        # myのみに色を付けて出力
                        if y == 0:
                            self.draw_room_placement(start_x=knot_pts[x][y][0], start_y=knot_pts[x][y][1], all=False,
                                                     now_agent=x,
                                                     font=font, font_size=font_size, color=color)
                        elif y == 1:
                            self.draw_q_bar(start_x=knot_pts[x][y][0] + int(self.img_contents_x_span/2),
                                            start_y=knot_pts[x][y][1], q=self.merged_q_temps[x][0], now_agent=now_agent,
                                            a_names=a_names, head_num=y - 2,
                                            font=font, font_size=font_size, color=color)
                        elif 1 < y:
                            self.draw_q_bar(start_x=knot_pts[x][y][0] + int(self.img_contents_x_span/2),
                                            start_y=knot_pts[x][y][1], q=self.agg_q_temps[x][y - 2][0],
                                            now_agent=now_agent,
                                            a_names=a_names, head_num=y - 2,
                                            font=font, font_size=font_size, color=color)
                            # もしagg_wなら赤枠を出力
                            if self.agg_w_temps[x][y - 2][0] == 5:
                                # cv2.rectangle(self.img, (knot_pts[x][y][0], knot_pts[x][y][1]),
                                #               (knot_pts[x+1][y+1][0], knot_pts[x+1][y+1][1]), (18, 0, 230),
                                #               thickness=1)
                                cv2.rectangle(self.img, tuple(knot_pts[x][y]), (tuple(knot_pts[x + 1][y + 1])),
                                              (18, 0, 230),
                                              thickness=1)
                    elif x == self.n_agents:
                        if y == 0:
                            # 全部のエージェントに色付けて出力
                            self.draw_room_placement(start_x=knot_pts[x][y][0], start_y=knot_pts[x][y][1], all=True, now_agent=x,
                                                     font=font, font_size=font_size, color=color)
                            # reward textの出力
                            self.draw_texts(knot_pts[x][y+1][0], knot_pts[x][y+1][1] + 15, now_agent, a_names, action, step,
                                            reward, reward_total, font, font_size, color)


            # 室配置，Q_headsの出力

            # for x in range(len(agg_q) + 2):
            #     for y in range(self.n_agents+1):
            #         if y < self.n_agents:
            #             # myのみに色を付けて出力
            #             if x == 0:
            #                 self.draw_room_placement(start_x=knot_pts[x][y][0], start_y=knot_pts[x][y][1], all=False,
            #                                          now_agent=x,
            #                                          font=font, font_size=font_size, color=color)
            #             elif x == 1:
            #                 self.draw_q_bar(start_x=knot_pts[x][y][0] + int(self.img_contents_x_span/2),
            #                                 start_y=knot_pts[x][y][1], q=self.merged_q_temps[x][0], now_agent=now_agent,
            #                                 a_names=a_names, head_num=y - 2,
            #                                 font=font, font_size=font_size, color=color)
            #             elif 1 < x:
            #                 self.draw_q_bar(start_x=knot_pts[x][y][0] + int(self.img_contents_x_span/2),
            #                                 start_y=knot_pts[x][y][1], q=self.agg_q_temps[x][y - 2][0],
            #                                 now_agent=now_agent,
            #                                 a_names=a_names, head_num=y - 2,
            #                                 font=font, font_size=font_size, color=color)
            #                 # もしagg_wなら赤枠を出力
            #                 if self.agg_w_temps[x][y - 2][0] == 5:
            #                     # cv2.rectangle(self.img, (knot_pts[x][y][0], knot_pts[x][y][1]),
            #                     #               (knot_pts[x+1][y+1][0], knot_pts[x+1][y+1][1]), (18, 0, 230),
            #                     #               thickness=1)
            #                     cv2.rectangle(self.img, tuple(knot_pts[x][y]), (tuple(knot_pts[x + 1][y + 1])),
            #                                   (18, 0, 230),
            #                                   thickness=1)
            #         elif y == self.n_agents:
            #             if x == 0:
            #                 # 全部のエージェントに色付けて出力
            #                 self.draw_room_placement(start_x=knot_pts[x][y][0], start_y=knot_pts[x][y][1], all=True, now_agent=x,
            #                                          font=font, font_size=font_size, color=color)
            #                 # reward textの出力
            #                 self.draw_texts(knot_pts[x][y+1][0], knot_pts[x][y+1][1] + 15, now_agent, a_names, action, step,
            #                                 reward, reward_total, font, font_size, color)

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

    def draw_grid(self, n_contents_x, n_contents_y, font, font_size, color):
        knot_points = [] # 全部バラバラ
        knot_points_y = []
        for x in range(self.img_contents_x_num):
            for y in range(self.img_contents_y_num):
                knot_points_y.append([int(x*self.img_contents_x_span), int(y*self.img_contents_y_span)])
            knot_points.append(knot_points_y)
            knot_points_y = []

        for point in (knot_points):
            for pt in point:
                cv2.circle(self.img, tuple(pt), 3, (200, 200, 200))
                # cv2.putText(self.img, str(tuple(pt)), tuple(pt), font, font_size, color)
                cv2.rectangle(self.img, (pt[0], pt[1]),
                              (int(pt[0]+self.img_contents_x_span), int(pt[1]+self.img_contents_y_span)), (70, 70, 70))

        return knot_points

    def draw_room_placement(self, start_x, start_y, all, now_agent, font, font_size, color):
        # siteの出力
        site_list = copy.deepcopy(self.site_0_list)

        # grid_pitchだけ先に掛ける
        site_arr = np.dot(np.array(site_list), self.img_grid_pitch)
        site_list = site_arr.tolist()

        # 取得したsite_listにスタート位置分の数値を足す
        for i, list in enumerate(site_list):
            site_list[i][0] = list[0] + start_y
            site_list[i][1] = list[1] + start_x

        # siteの出力
        for i, list in enumerate(site_list):
            cv2.rectangle(self.img,
                          (int(list[:][1] + self.img_grid_pitch),
                           int(list[:][0])),
                          (int(list[:][1]),
                           int(list[:][0] + self.img_grid_pitch)),
                          (77, 77, 77),
                          thickness=-1)
            cv2.rectangle(self.img,
                          (int(list[:][1] + self.img_grid_pitch),
                           int(list[:][0])),
                          (int(list[:][1]),
                           int(list[:][0] + self.img_grid_pitch)),
                          (200, 200, 200))

        # agentのindex作成
        agent_indexes = []
        for i in range(self.n_agents):
            agent_index = np.array(np.where(self.state_t == i)).T.tolist()
            # agent_indexにgrid_pitchを掛ける
            agent_index_arr = np.dot(np.array(agent_index), self.img_grid_pitch)
            agent_index = agent_index_arr.tolist()
            agent_indexes.append(agent_index)

        # 取得したindexにスタート位置分の数値を足す
        for agent_num, indexes in enumerate(agent_indexes):
            for i, index in enumerate(indexes):
                agent_indexes[agent_num][i][0] = index[0] + start_y
                agent_indexes[agent_num][i][1] = index[1] + start_x

        # agentの出力
        for i, list in enumerate(agent_indexes):
            if len(list) == 1:  # agentが1マス
                if all:  # 全エージェントの色を塗る
                    cv2.rectangle(self.img,
                                  (int(list[0][1] + self.img_grid_pitch), int(list[0][0])),
                                  (int(list[0][1]), int(list[0][0] + self.img_grid_pitch)),
                                  self.number_to_color(i), thickness=-1)

                else:
                    if i == now_agent:
                        cv2.rectangle(self.img,
                                      (int(list[0][1] + self.img_grid_pitch), int(list[0][0])),
                                      (int(list[0][1]), int(list[0][0] + self.img_grid_pitch)),
                                      self.number_to_color(i), thickness=-1)
                    else:
                        cv2.rectangle(self.img,
                                      (int(list[0][1] + self.img_grid_pitch), int(list[0][0])),
                                      (int(list[0][1]), int(list[0][0] + self.img_grid_pitch)),
                                      (100, 100, 100), thickness=-1)

                cv2.rectangle(self.img,
                              (int(list[0][1] + self.img_grid_pitch), int(list[0][0])),
                              (int(list[0][1]), int(list[0][0] + self.img_grid_pitch)),
                              (200, 200, 200))
            else:  # agentが複数マス
                for j, j_list in enumerate(list):
                    if all:  # 全エージェントの色を塗る
                        cv2.rectangle(self.img,
                                      (int(j_list[:][1] + self.img_grid_pitch), int(j_list[:][0])),
                                      (int(j_list[:][1]), int(j_list[:][0] + self.img_grid_pitch)),
                                      self.number_to_color(i), thickness=-1)
                    else:
                        if i == now_agent:
                            cv2.rectangle(self.img,
                                          (int(j_list[:][1] + self.img_grid_pitch), int(j_list[:][0])),
                                          (int(j_list[:][1]), int(j_list[:][0] + self.img_grid_pitch)),
                                          self.number_to_color(i), thickness=-1)
                        else:
                            cv2.rectangle(self.img,
                                          (int(j_list[:][1] + self.img_grid_pitch), int(j_list[:][0])),
                                          (int(j_list[:][1]), int(j_list[:][0] + self.img_grid_pitch)),
                                          (100, 100, 100), thickness=-1)

                    cv2.rectangle(self.img,
                                  (int(j_list[:][1] + self.img_grid_pitch), int(j_list[:][0])),
                                  (int(j_list[:][1]), int(j_list[:][0] + self.img_grid_pitch)),
                                  (200, 200, 200))

        # textの出力　室配置の中心下に
        if all:
            cv2.putText(self.img, 'all_agents',
                        (start_x, start_y + (self.col+2) * self.img_grid_pitch),
                        font, font_size, color)
        else:
            cv2.putText(self.img, 'agent: ' + str(now_agent),
                        (start_x, start_y + (self.col+2) * self.img_grid_pitch),
                        font, font_size, color)

        # 室エージェントの中心indexである座標値を取得
        my_center_list = []
        for agent_num, agent in enumerate(agent_indexes):
            center_agent = agent[len(agent) // 2]
            my_center_list.append(center_agent)

        # your_agentの配列をコピー
        your_agent_arr = np.array(self.your_agent[:])
        # your_centerの配列　shape=(4, 4, 1)
        your_center_arr = np.zeros((your_agent_arr.shape[0], your_agent_arr.shape[1], 2))

        # your_agentのyour番号をそれぞれの座標に変更
        for i in range(self.n_agents):
            your_agents = np.array(np.where(your_agent_arr == i)).T.tolist()
            for your_agent in your_agents:
                # your_center_arr[your_agent[0], your_agent[1], 0] = my_center_list[i]
                your_center_arr[your_agent[0]][your_agent[1]] = my_center_list[i]

        # my_centerからyour_centerに向けて線を引く．
        for i, my_center in enumerate(my_center_list):
            for your_center in your_center_arr[i]:
                if int(your_center[0]) is not 0:
                    # cv2.line(self.img,
                    #          (int(my_center[1] * self.img_grid_pitch), int(my_center[0] * self.img_grid_pitch)),
                    #          (int(your_center[1] * self.img_grid_pitch), int(your_center[0] * self.img_grid_pitch)),
                    #          (255, 255, 255))
                    cv2.line(self.img,
                             (int(my_center[1]), int(my_center[0])),
                             (int(your_center[1]), int(your_center[0])),
                             (255, 255, 255))

    def draw_q_bar(self, start_x, start_y, q, now_agent, a_names, head_num, font, font_size, color):
        bar_span = 12  # バー同士の間隔
        bar_col_w = 5  # 縦バーの幅
        bar_row_w = 8  # 横バーの幅
        bar_row_h = 100/max(q)  # 横バーの長さを決める係数 qのmaxから決める．
        max_a_num = np.argmax(np.array(q)) # Qの最大値インデックス

        bar_col_h = len(self.enable_actions) * bar_span  # 縦バーの長さ
        # action_name = action_names[action]

        # qの縦バー
        # cv2.rectangle(self.img,
        #               (start_x, start_y),
        #               (start_x + bar_col_w, start_y + bar_col_h),
        #               (200, 200, 200), thickness=-1)
        # qの横バー
        for i in range(len(self.enable_actions)):
            if i == max_a_num:
                cv2.rectangle(self.img,
                              (int(start_x+bar_col_w-20), int(start_y + (i * bar_span))),
                              (int(start_x + (q[i] * bar_row_h)), int(start_y + (i * bar_span) + bar_row_w)),
                              (18, 0, 230), thickness=-1)
                if self.limit_flag:
                    cv2.putText(self.img, a_names[i], (int(start_x - 100), (int(start_y + (i * bar_span) + bar_row_w))),
                                font, font_size, (18, 0, 230))
                else:
                    cv2.putText(self.img, a_names[i], (int(start_x - 100), (int(start_y + (i * bar_span) + bar_row_w))),
                                font, font_size, color)
                cv2.putText(self.img, str(round(q[i], 2)),
                            (start_x + bar_col_w - 17, (int(start_y + (i * bar_span) + bar_row_w))),
                            font, font_size, (0, 0, 0))
            else:
                cv2.rectangle(self.img,
                              (int(start_x+bar_col_w-20), int(start_y+(i*bar_span))),
                              (int(start_x + (q[i]*bar_row_h)), int(start_y+(i*bar_span)+bar_row_w)),
                              (200, 200, 200), thickness=-1)
                cv2.putText(self.img, a_names[i], (int(start_x - 100), (int(start_y + (i * bar_span) + bar_row_w))),
                            font, font_size, color)
                cv2.putText(self.img, str(round(q[i], 2)),
                            (start_x + bar_col_w - 17, (int(start_y + (i * bar_span) + bar_row_w))),
                            font, font_size, (0, 0, 0))

        # ヘッドの名前
        if head_num==-1:
            cv2.putText(self.img, 'head_name: merged',
                        (start_x - 50, start_y + self.img_contents_y_span - 10),
                        font, font_size, color)

        else:
            cv2.putText(self.img, 'head_name: ' + str(self.reward_name[head_num]),
                        (start_x - 50, start_y + self.img_contents_y_span - 10),
                        font, font_size, color)


    def draw_texts(self, start_x, start_y, now_agent, a_names, action, step, reward, reward_total, font, font_size, color):
        action_name = a_names[action]
        cv2.putText(self.img, 'agent: ' + str(now_agent), (start_x, start_y), font, font_size, color)
        if self.limit_flag:
            cv2.putText(self.img, 'action: ' + str(action_name), (start_x, start_y + 15), font, font_size, (18, 0, 230))
        else:
            cv2.putText(self.img, 'action: ' + str(action_name), (start_x, start_y + 15), font, font_size, color)
        cv2.putText(self.img, 'step: ' + str(step), (start_x, start_y + 30), font, font_size, color)
        cv2.putText(self.img, 'reward: ' + str(reward), (start_x, start_y + 45), font, font_size, color)
        cv2.putText(self.img, 'reward_total: ' + str(reward_total), (start_x, start_y + 60), font, font_size, color)

    # def gif_animation(self, episode):
    #     if episode % self.draw_movie_freq == 0 and episode != 0:
    #         folderName = "./cv2_image/images"
    #
    #         # 画像ファイルの一覧を取得
    #         picList = glob.glob(folderName + "\*.png")
    #
    #         # figオブジェクトを作る
    #         fig = plt.figure()
    #
    #         # 空のリストを作る
    #         ims = []
    #
    #         # 画像ファイルを順々に読み込んでいく
    #         for i in range(len(picList)):
    #             # 1枚1枚のグラフを描き、appendしていく
    #             tmp = Image.open(picList[i])
    #             tmp = np.asarray(tmp)
    #             ims.append([plt.imshow(tmp, animated=True)])
    #
    #             # アニメーション作成
    #         ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True)
    #         # ani.save("./cv2_image/gif_ani/test.gif")
    #         dt = datetime.now().strftime("%m%d_%H%M")
    #         # ani.save('./cv2_image/1022_0043/animation' +str(dt) + '.gif', writer="imagemagick")
    #         ani.save('./cv2_image/gif_ani/animation' +str(dt) + '.gif', writer="imagemagick")

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
        return self.state_channel_t, self.next_state_channel_t, self.reward, self.reward_channels, self.game_over, self.term
        # return self.state_4chan_t, self.next_state_4chan_t, self.reward, self.reward_channels, self.game_over

    def execute_action(self, action, now_agent):
        self.step(action, now_agent)
        # self.step_deform(action, now_agent)

    def reset(self):
        self.reward = 0
        self.reward_channels = np.zeros(len(self.reward_scheme), dtype=np.float32)
        self.game_over = False
        self.term = False
        self.step_id = 0

        # 現エージェントの状態の初期化
        # self.state_t = self.state_0

        if self.evely_random_flag:
            self.state_seed_0 = self.init_state_seed()
            self.state_0 = self.init_state()
        else:
            # self.state_0 = self.init_state()
            pass

        # self.state_t = self.init_state()
        # self.state_4chan_t = self.state_4chan_0
        self.state_channel_t = self.state_channel_0

        # 次エージェントの状態の初期化
        # self.next_state_4chan_t = self.next_state_4chan_0
        self.next_state_channel_t = self.next_state_channel_0

