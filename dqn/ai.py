import numpy as np
from collections import deque
from keras import backend as K
from keras.optimizers import RMSprop
import tensorflow as tf
from copy import deepcopy


from utils import ExperienceReplay, flatten, slice_tensor_tensor
from dqn.model import build_dense, build_cnn
import dqn.experiment as expt
from environment.room_placement import RoomPlacement

floatX = 'float32'

class AI:
    def __init__(self, state_shape, nb_actions, action_dim, reward_dim, history_len=1, gamma=.99,
                 learning_rate=0.00025, epsilon=0.05, final_epsilon=0.05, test_epsilon=0.001,
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

        # ここでreplay memory を保持
        # self.D = deque(maxlen=self.replay_memory_size)
        # self.temp_D = deque(maxlen=self.replay_memory_size)
        # self.temp_rot_D =deque(maxlen=self.replay_memory_size)

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
                cost = self._compute_cost(qs[-1], a, r[:, 1], t, q2s[-1])
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
        
    def get_max_action(self, states):
        # stateのreshape: 未実装
        # states = self._reshape(states)
        q = np.array(self.predict_network([states]))
        q = np.sum(q, axis=0)
        return np.argmax(q, axis=1)
    
    def get_action(self, states, evaluate):
        eps = self.epsilon if not evaluate else self.test_epsilon
        if self.rng.binomial(1, eps):
            return self.rng.randint(self.nb_actions)
        else:
            return self.get_max_action(states=states)






    def store_exp(self, reward_cnt, average_reward_cnt):
        if reward_cnt > 100 and reward_cnt > average_reward_cnt:
            # temp_Dのデータを回転してtemp_Dに格納する
            # self.rotate_experience()
            # temp_DをDに格納する。
            self.D.extend(self.temp_D)
            print('store, exp_num: ' + str(len(self.D)))

        # temp_Dの消去
        self.temp_D.clear()
        self.temp_rot_D.clear()

        # return (len(self.D) >= self.replay_memory_size)
        return (len(self.D) >= self.replay_start)
    
    def store_temp_exp(self, states, action, reward, states_1, terminal):
        # self.temp_D.append((states, action, reward, states_1, terminal))
        self.temp_D.append(np.array(states, action, reward, states_1, terminal))
    
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

    def update_exploration(self):
        if self.epsilon > FINAL_EXPLORATION:
            # self.exploration -= self.exploration_step * num
            self.epsilon -= self.epsilon_step * 1
            if self.epsilon < FINAL_EXPLORATION:
                self.expsilon = FINAL_EXPLORATION
        
        print(self.epsilon)
        
    def experience_replay(self):
        state_minibatch = []
        y_minibatch = []
        action_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        # storeからランダムにミニバッチのインデックスを指定して取得．
        # それらをminibatch_sizeまでminibatchのリストにappendしていく
        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = self.enable_actions.index(action_j)

            # state_jを入力として、modelから推論されるQ_values
            # y_j = self.Q_values(state_j)
            y_j = self.predict_network([state_j])

            # Q値の更新式．教師信号y_trueになる．
            if terminal:
                # terminalの扱い．ゴールにたどり着き次のステップがない，という状態であれば，そのまま報酬がQとなる
                y_j[action_j_index] = reward_j
                # v = np.max(self.Q_values(state_j_1, isTarget=True))
                # y_j[action_j_index] = reward_j + self.discount_factor * v
            else:
                # v = self.Q_values(state_j_1, isTarget=True)[action_j_index]
                # target_modelより推論されるQ_valuesの最大値からvを計算。
                v = np.max(self.predict_target_network([state_j_1]))
                # 実際に行動として選択したノードのQを更新　
                y_j[action_j_index] = reward_j + self.gamma * v

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)
            action_minibatch.append(action_j_index)

        validation_data = None
    

        



    @staticmethod
    def weight_transfer(from_model, to_model):
        for f_model, t_model in zip(from_model, to_model):
            t_model.set_weights(deepcopy(f_model.get_weights()))