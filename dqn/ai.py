import time
from copy import deepcopy

import numpy as np
from keras import backend as K
from keras.optimizers import RMSprop

from dqn.model import build_cnn
from utils import ExperienceReplay, flatten, slice_tensor_tensor

floatX = 'float32'

class AI:
    def __init__(self, state_shape, nb_actions, action_dim, reward_dim, history_len=1, gamma=.99,
                 learning_rate=0.00025, annealing=True, annealing_episodes=5000, epsilon=1.0, final_epsilon=0.05, test_epsilon=0.001,
                minibatch_size=32, replay_max_size=100, replay_memory_size=50000,
                 update_freq=50, learning_frequency=1,
                 num_units=250, remove_features=False, use_mean=False, use_hra=True, rng=None):
        self.rng = rng
        self.history_len = history_len
        # self.state_shape = [1] + state_shape # この操作が謎　
        self.state_shape = state_shape
        self.nb_actions = nb_actions
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learning_rate_start = learning_rate

        self.epsilon = epsilon
        self.start_epsilon = epsilon
        self.test_epsilon = test_epsilon
        self.final_epsilon = final_epsilon
        self.annealing = annealing
        self.annealing_episodes = annealing_episodes
        self.annealing_episode = (self.start_epsilon - self.final_epsilon) / self.annealing_episodes

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
        print('Compiled Model and Learning.')


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
                cost = self._compute_cost(qs[-1], a, r[:, i], t, q2s[-1])
                optimizer = RMSprop(lr=self.learning_rate, rho=.95, epsilon=1e-7)

                # 学習設定
                updates += optimizer.get_updates(params=self.networks[i].trainable_weights, loss=cost)
                # self.networks[i].compile(loss=cost, optimizer=optimizer)
                costs += cost

        # target_netのweightを更新　
        target_updates = []
        for network, target_network in zip(self.networks, self.target_networks):
            for target_weight, network_weight in zip(target_network.trainable_weights, network.trainable_weights):
                target_updates.append(K.update(target_weight, network_weight)) # from, to

        # kerasの関数のインスタンスを作成　updates: 更新する命令のリスト．
        self._train_on_batch = K.function(inputs=[s, a, r, s2, t], outputs=[costs], updates=updates)
        self.predict_network = K.function(inputs=[s], outputs=qs)
        self.update_weights = K.function(inputs=[], outputs=[], updates=target_updates)

    def update_epsilon(self):
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.annealing_episode * 1
            if self.epsilon < self.final_epsilon:
                self.epsilon = self.final_epsilon

    def get_max_action(self, states):
        # stateのreshape: 未実装
        states = np.expand_dims(states, axis=0)
        q = np.array(self.predict_network([states]))
        q = np.sum(q, axis=0)
        return np.argmax(q, axis=1)
    
    def get_action(self, states, evaluate):
        if not evaluate:
            eps = self.epsilon
        else:
            eps = self.test_epsilon

        if self.rng.binomial(1, eps):
            return self.rng.randint(self.nb_actions)
        else:
            return self.get_max_action(states=states)[0]

    def train_on_batch(self, s, a, r, s2, t):
        # 元コード　expand_dimsをしている
        # s = self._reshape(s)
        # s2 = self._reshape(s2)
        # if len(r.shape) == 1:
        #     r = np.expand_dims(r, axis=-1)

        return self._train_on_batch([s, a, r, s2, t])

    def learn(self):
        start_time = time.time()

        assert self.minibatch_size <= len(self.transitions.D), 'not enough data in the pool'

        # 経験のサンプリング
        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)

        objective = self.train_on_batch(s, a, r, s2, term)

        # ターゲットに対してネットワークの更新
        if self.update_counter == self.update_freq:
            self.update_weights([])
            self.update_counter = 0
        else:
            self.update_counter += 1

        learn_time = time.time() - start_time

        return objective, learn_time

    def dump_network(self, weights_file_path='q_network_weights.h5', overwrite=True):
        for i, network in enumerate(self.networks):
            network.save_weights(weights_file_path[:-3] + str(i) + weights_file_path[-3:], overwrite=overwrite)

    @staticmethod
    def weight_transfer(from_model, to_model):
        for f_model, t_model in zip(from_model, to_model):
            t_model.set_weights(deepcopy(f_model.get_weights()))