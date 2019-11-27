import time
from copy import deepcopy

import numpy as np
import tensorflow as tf
from keras.backend import tensorflow_backend as K
# from keras import backend as K
from keras.optimizers import RMSprop

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
# config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1))
session = tf.Session(config=config)
K.set_session(session)


from dqn.model import build_cnn
from utils import ExperienceReplay, flatten, slice_tensor_tensor

floatX = 'float32'

class AI:
    def __init__(self, state_shape, nb_actions, action_dim, reward_dim, history_len=1, gamma=.99, is_aggregator=True,
                 learning_rate=0.00025, transfer_lr=0.0001, final_lr=0.001, annealing_lr=True, annealing=True, annealing_episodes=5000, epsilon=1.0, final_epsilon=0.05, test_epsilon=0.001,
                minibatch_size=32, replay_max_size=100, replay_memory_size=50000,
                 update_freq=50, learning_frequency=1,
                 num_units=250, remove_features=False, use_mean=False, use_hra=True, rng=None, test=False, transfer_learn=False):
        self.test = test
        self.transfer_learn = transfer_learn

        self.rng = rng
        self.history_len = history_len
        # self.state_shape = [1] + state_shape # この操作が謎　
        self.state_shape = state_shape
        self.nb_actions = nb_actions
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.gamma = gamma

        self.is_aggregator = is_aggregator
        self.agg_w = np.ones((self.reward_dim, 1, 1))

        self.qs = np.zeros((self.reward_dim, 1, self.nb_actions))
        self.agg_q = np.zeros((self.reward_dim, 1, self.nb_actions))
        self.merged_q = np.zeros((1, self.nb_actions))
        self.qs_list = []
        self.agg_q_list = []
        self.merged_q_list = []

        self.epsilon = epsilon
        self.start_epsilon = epsilon
        self.test_epsilon = test_epsilon
        self.final_epsilon = final_epsilon
        self.annealing = annealing
        self.annealing_episodes = annealing_episodes
        self.annealing_episode = (self.start_epsilon - self.final_epsilon) / self.annealing_episodes

        if not self.transfer_learn:
            self.learning_rate = learning_rate
            self.start_lr = learning_rate
        else:
            self.learning_rate = transfer_lr
            self.start_lr = transfer_lr
        self.final_lr = final_lr
        self.annealing_lr = annealing_lr
        self.annealing_episode_lr = (self.start_lr - self.final_lr) / self.annealing_episodes

        self.get_action_time_channel = np.zeros(4)
        self.get_max_a_time_channel = np.zeros(3)

        self.minibatch_size = minibatch_size
        self.update_freq = update_freq
        self.update_counter = 0
        self.nb_units = num_units
        self.use_mean = use_mean
        self.use_hra = use_hra
        self.remove_features = remove_features
        self.learning_frequency = learning_frequency
        self.replay_max_size = replay_max_size
        self.replay_memory_size = replay_memory_size

        self.transitions = ExperienceReplay(max_size=self.replay_max_size, history_len=history_len, rng=self.rng,
                                            state_shape=state_shape, action_dim=action_dim, reward_dim=reward_dim)

        # ネットワークの構築
        self.networks = [self._build_network() for _ in range(self.reward_dim)]
        self.target_networks = [self._build_network() for _ in range(self.reward_dim)]

        # パラメータの保持 reward_dim個のネットワークにある各層の重みをflatten
        self.all_params = flatten([network.trainable_weights for network in self.networks])
        self.all_target_params = flatten([target_network.trainable_weights for target_network in self.target_networks])

        # target_networksの重みを更新する．
        self.weight_transfer(from_model=self.networks, to_model=self.target_networks)

        # ネットワークのコンパイル lossなどの定義
        self._compile_learning()
        if not self.test:
            if self.transfer_learn:
                self.load_weights(weights_file_path='./learned_weights/init_weights_7chan/q_network_weights.h5')
                print('Compiled Model. -- Transfer Learning -- ')
                print('learning rate: ' + str(self.learning_rate))
            else:
                print('Compiled Model. -- Learning -- ')

        else:
            # self.load_weights(weights_file_path='./results/test_weights/q_network_weights.h5')
            # self.load_weights(weights_file_path='./learned_weights/test_weights_7chan/q_network_weights.h5')
            self.load_weights(weights_file_path='./learned_weights/test_weights_7chan_8room/q_network_weights.h5')


            print('Compiled Model and Load weights. -- Testing -- ')


    def _build_network(self):
        # model.build_dense　→　浅いニューラルネットを構築
        # model.build_cnn　→　CNNを構築

        return build_cnn(self.state_shape, int(self.nb_units / self.reward_dim),
                           self.nb_actions, self.reward_dim, self.remove_features)

    def _compute_cost(self, q, a, r, t, q2):
        preds = slice_tensor_tensor(q, a)
        bootstrap = K.max if not self.use_mean else K.mean
        targets = r + (1 - t) * self.gamma * bootstrap(q2, axis=1)
        cost = K.sum((targets - preds) ** 2)
        return cost

    def _compute_cost_huber(self, q, a, r, t, q2):
        preds = slice_tensor_tensor(q, a)
        bootstrap = K.max if not self.use_mean else K.mean
        targets = r + (1 - t) * self.gamma * bootstrap(q2, axis=1)
        err = targets - preds
        cond = K.abs(err) > 1.0
        L2 = 0.5 * K.square(err)
        L1 = (K.abs(err)-0.5)
        cost = tf.where(cond, L2, L1)
        return K.mean(cost)


    def _compile_learning(self):
        # ミニバッチの状態で入力できるようにするplaceholder

        # s = K.placeholder(shape=tuple([None] + [self.history_len] + self.state_shape)) # history?
        s = K.placeholder(shape=tuple([None] + self.state_shape))
        a = K.placeholder(ndim=1, dtype='int32')
        r = K.placeholder(ndim=2, dtype='float32')
        # s2 = K.placeholder(shape=tuple([None] + [self.history_len] + self.state_shape))
        s2 = K.placeholder(shape=tuple([None] + self.state_shape))
        t = K.placeholder(ndim=1, dtype='float32')

        updates = []
        costs = 0
        # costs_arr = np.zeros(len(self.networks))
        costs_list = []
        qs = []
        q2s = []

        # 構築したネットワーク分だけ処理
        for i in range(len(self.networks)):
            local_s =s
            local_s2 = s2

            # remove_features　→　未実装

            # 推論値　s: Stをinputとして
            qs.append(self.networks[i](local_s))
            # 教師値 s: St+1をinputとして
            q2s.append(self.target_networks[i](local_s2))

            if self.use_hra:
                # cost = lossの計算
                # cost = self._compute_cost(qs[-1], a, r[:, i], t, q2s[-1])
                cost = self._compute_cost(qs[-1], a, r[:, i], t, q2s[-1])

                optimizer = RMSprop(lr=self.learning_rate, rho=.95, epsilon=1e-7)

                # 学習設定
                updates += optimizer.get_updates(params=self.networks[i].trainable_weights, loss=cost)
                # self.networks[i].compile(loss=cost, optimizer=optimizer)
                # costの合計
                costs += cost
                # 各costが格納されたリスト
                costs_list.append(cost)
                # costs_arr[i] = cost

        # target_netのweightを更新　
        target_updates = []
        for network, target_network in zip(self.networks, self.target_networks):
            for target_weight, network_weight in zip(target_network.trainable_weights, network.trainable_weights):
                target_updates.append(K.update(target_weight, network_weight)) # from, to

        # kerasの関数のインスタンスを作成　updates: 更新する命令のリスト．
        # self._train_on_batch = K.function(inputs=[s, a, r, s2, t], outputs=[costs], updates=updates)
        self._train_on_batch = K.function(inputs=[s, a, r, s2, t], outputs=costs_list, updates=updates)
        self.predict_network = K.function(inputs=[s], outputs=qs)
        self.predict_target_network = K.function(inputs=[s], outputs=qs)
        self.update_weights = K.function(inputs=[], outputs=[], updates=target_updates)

    def update_epsilon(self):
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.annealing_episode * 1
            if self.epsilon < self.final_epsilon:
                self.epsilon = self.final_epsilon

    def update_lr(self):
        if self.annealing_lr:
            if self.learning_rate > self.final_lr:
                self.learning_rate -= self.annealing_episode_lr * 1
                if self.learning_rate < self.final_lr:
                    self.learning_rate = self.final_lr

    def get_max_action(self, states):
        # stateのreshape: 未実装
        # start = time.time()
        states = np.expand_dims(states, axis=0)
        # expand_dim_time = round(time.time() - start, 8)

        # start = time.time()
        self.qs = np.array(self.predict_network([states]))
        # predict_q_time = round(time.time() - start, 8)

        # print(q)
        # print(self.agg_w)
        # aggのweightを掛ける

        # start = time.time()
        self.agg_q = self.qs * self.agg_w
        # print(q)
        self.merged_q = np.sum(self.agg_q, axis=0)
        # agg_w_time = round(time.time() - start, 8)

        # self.get_max_a_time_channel = [expand_dim_time, predict_q_time, agg_w_time]
        return np.argmax(self.merged_q, axis=1)

    def get_action(self, states, evaluate, pre_reward_channels):
        start = time.time()
        if not evaluate:
            eps = self.epsilon
        else:
            eps = self.test_epsilon
        epsilon_time = round(time.time() - start, 8)

        start = time.time()
        self.aggregator(pre_reward_channels)
        aggregator_time = round(time.time() - start, 8)

        start = time.time()
        self.rng.binomial(1, eps)
        rng_time = round(time.time() - start, 8)

        start = time.time()
        # a = self.get_max_action(states=states)[0]
        max_action_time = round(time.time() - start, 8)

        self.get_action_time_channel = [epsilon_time, aggregator_time, rng_time, max_action_time]

        # εグリーディ
        if self.rng.binomial(1, eps):
            return self.rng.randint(self.nb_actions)
        else:
            return self.get_max_action(states=states)[0]
            # return self.rng.randint(self.nb_actions)

    def aggregator(self, reward_channels):

        if self.is_aggregator:
            # 単数接続用のagg
            if self.state_shape[0] == 4:
                if reward_channels[0] < 1.0:
                    self.agg_w[0][0][0] = 5 # connect
                    self.agg_w[1][0][0] = 1 # shape
                    self.agg_w[2][0][0] = 1 # area
                else:
                    self.agg_w[0][0][0] = 1
                    self.agg_w[1][0][0] = 5
                    self.agg_w[2][0][0] = 5

            # 複数接続用のagg
            elif self.state_shape[0] == 7:
                # 接続報酬のインデックス
                connect_heads = reward_channels[0:4]

                connect_num = sum(1 for i in connect_heads if not np.isnan(i))
                connect_reward = sum(i for i in connect_heads if not np.isnan(i))

                # 接続条件を満たしていない場合　→　接続の報酬が　接続の最大報酬になっていない場合
                if connect_num * 1.0 != round(connect_reward, 1):
                    for index, reward in enumerate(reward_channels):
                        # 接続報酬
                        if 0<=index<=3:
                            if reward == 1.0: # 接続している
                                self.agg_w[index][0][0] = 1
                            elif reward <= 0.0: # 接続していない　もしくは　衝突
                                self.agg_w[index][0][0] = 5
                            elif np.isnan(reward): # 接続相手がない
                                self.agg_w[index][0][0] = 0.1

                        # # 衝突報酬
                        # elif index == 4:
                        #     self.agg_w[index][0][0] = 5

                        # 面積，形状報酬，有効寸法
                        else:
                            self.agg_w[index][0][0] = 1

                # 接続条件を満たしている場合
                else:
                    for index, reward in enumerate(reward_channels):
                        # 接続報酬
                        if 0<=index<=3:
                            if reward == 1.0: # 接続している
                                self.agg_w[index][0][0] = 1
                            elif reward <= 0.0: # 接続していない　もしくは　衝突
                                self.agg_w[index][0][0] = 1
                            elif np.isnan(reward): # 接続相手がない
                                self.agg_w[index][0][0] = 0.1

                        # # 衝突報酬
                        # elif index == 4:
                        #     self.agg_w[index][0][0] = 1

                        # 面積，形状報酬，有効寸法
                        else:
                            self.agg_w[index][0][0] = 5

        else:
            # raise ValueError("not use aggregator")
            pass

    def get_TDerror(self):
        sum_TDerror = 0
        s, a, r, s2, t = self.transitions.temp_D[len(self.transitions.temp_D)-1]
        a = [a]
        a2 = self.get_max_action(s2) # t+1での最大行動

        s = np.expand_dims(s, axis=0)
        s2 = np.expand_dims(s2, axis=0)

        for i in range(len(self.networks)):
            # 各headでTD errorを計算して，それをsum
            target = r[i] + self.gamma * np.array(self.predict_target_network([s2]))[i][0][a2][0] # target_netから
            TDerror = target - np.array(self.predict_target_network([s]))[i][0][a][0]
            sum_TDerror += TDerror

        return sum_TDerror

    def update_TDerror(self):
        for i in range(0, len(self.transitions.D)-1):
            (s, a, r, s2) = self.transitions.D[i]
            a2 = self.get_max_action(s2)
            target = r + self.gamma * self.predict_target_network([s2])[a2]
            TDerror = target - self.predict_target_network([s])[a]
            self.transitions.TDerror_buffer[i] = TDerror

    def get_sum_abs_TDerror(self):
        sum_abs_TDerror = 0
        for i in range(0, len(self.transitions.D)-1):
            sum_abs_TDerror += abs(self.transitions.TDerror_buffer[i]) + 0.0001 # 最新の状態データを取得

        return sum_abs_TDerror

    def train_on_batch(self, s, a, r, s2, t):
        # 元コード　expand_dimsをしている
        # s = self._reshape(s)
        # s2 = self._reshape(s2)
        # if len(r.shape) == 1:
        #     r = np.expand_dims(r, axis=-1)

        # minibatch分だけ入力
        return self._train_on_batch([s, a, r, s2, t])

    def learn(self):
        start_time = time.time()

        assert self.minibatch_size <= len(self.transitions.D), 'not enough data in the pool'

        # 経験のサンプリング
        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)

        cost_channel = self.train_on_batch(s, a, r, s2, term)
        if not isinstance(cost_channel, (list)):
            cost_channel = np.zeros(len(self.networks))

        # ターゲットに対してネットワークの更新
        if self.update_counter == self.update_freq:
            self.update_weights([])
            self.update_counter = 0
        else:
            self.update_counter += 1

        learn_time = time.time() - start_time

        return cost_channel, learn_time

    def prioritized_exp_replay(self):
        sum_abs_TDerror = self.get_sum_abs_TDerror()
        generatedrand_list = np.random.uniform(0, sum_abs_TDerror, self.minibatch_size)
        generatedrand_list = np.sort(generatedrand_list)


    def dump_network(self, weights_file_path='q_network_weights.h5', overwrite=True):
        for i, network in enumerate(self.networks):
            network.save_weights(weights_file_path[:-3] + str(i) + weights_file_path[-3:], overwrite=overwrite)

    def load_weights(self, weights_file_path='q_network_weights.h5'):
        for i, network in enumerate(self.networks):
            network.load_weights(weights_file_path[:-3] + str(i) + weights_file_path[-3:])
        self.update_weights([])

    @staticmethod
    def weight_transfer(from_model, to_model):
        for f_model, t_model in zip(from_model, to_model):
            t_model.set_weights(deepcopy(f_model.get_weights()))