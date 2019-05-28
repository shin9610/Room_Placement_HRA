import time
import numpy as np
from utils import Font, plot_and_write, create_folder, compute_ave, graph
import copy

class DQNExperiment(object):
    def __init__(self, env, ai, episode_max_len, history_len=1, max_start_nullops=1, replay_min_size=0,
                 score_window_size=100, rng=None, draw_graph_freq=100, folder_location='/experiments/',
                 folder_name='expt', testing=False):
        self.rng = rng
        self.fps = 0
        self.episode_num = 0
        self.step_num = 0
        self.last_episode_steps = 0
        self.total_training_steps = 0
        self.score = 0
        self.temp_score = 0
        self.ave_score = 0
        self.eval_steps = []
        self.eval_scores = []
        self.env = env
        self.ai = ai
        self.history_len = history_len
        self.max_start_nullops = max_start_nullops
        if not testing:
            self.folder_name = create_folder(folder_location, folder_name)
        self.episode_max_len = episode_max_len
        self.num_agents = env.n_agents
        self.score_window = np.zeros(score_window_size)
        self.steps_agent_window = np.zeros(score_window_size)
        self.replay_min_size = max(self.ai.minibatch_size, replay_min_size)
        self.last_state = np.empty(tuple([self.history_len] + self.env.state_shape), dtype=np.uint8)

        self.draw_graph_freq = draw_graph_freq


    def do_training(self, total_eps=5000, eps_per_epoch=100, eps_per_test=100, is_learning=True, is_testing=True):
        # total eps に達するまでepsを行う
        scores = []

        while self.episode_num < total_eps:
            print(Font.yellow + Font.bold + 'Training ... ' + str(self.episode_num) + '/' + str(total_eps) + Font.end,
                  end='\n')
            self.do_episodes(number=eps_per_epoch, is_learning=is_learning)
            scores.append(self.score)
            graph(self.episode_num, scores, self.draw_graph_freq)

            # if is_testing:
            #     eval_scores, eval_steps = self.do_episodes(number=eps_per_test, is_learning=False)
            #     self.eval_steps.append(eval_steps)
            #     self.eval_scores.append(eval_scores)
            #     plot_and_write(plot_dict={'steps': self.eval_steps}, loc=self.folder_name + "/steps",
            #                    x_label="Episodes", y_label="Steps", title="", kind='line', legend=True)
            #     plot_and_write(plot_dict={'scores': self.eval_scores}, loc=self.folder_name + "/scores",
            #                    x_label="Episodes", y_label="Scores", title="", kind='line', legend=True)
            #     self.ai.dump_network(weights_file_path=self.folder_name + '/q_network_weights.h5',
            #                          overwrite=True)

    def do_episodes(self, number=1, is_learning=True):
        scores = []
        
        # eps中において，eps per epochに達するまでepsを行う　→　1回に指定
        for num in range(number):
            self._do_episode(is_learning=is_learning)

            # 学習が終了したなら
            if not is_learning:
                # 元コードの処理が不明　→　passで
                # self.score_window = self._update_window(self.score_window, self.score)
                # self.steps_agent_window = self._update_window(self.steps_agent_window, self.last_episode_steps)
                pass
            else:
                # episode_numを足す
                self.episode_num += 1
        return np.mean(scores)

    def _do_episode(self, is_learning=True, evaluate=False):
        rewards = []

        self.env.reset()
        self._reset()
        game_over = False

        start_time = time.time()

        # 環境の観測
        state_t_1, next_state_t_1, reward_t, reward_channels, game_over = self.env.observe()

        # eps中において，termに達するまでstepを行う
        while not game_over:
            # 各agent順に行動
            for now_agent in range(self.env.n_agents):
                self.step_num += 1

                self.last_episode_steps += 1
                state_t = copy.deepcopy(next_state_t_1)

                # 行動選択　→　last_stateをどこから持ってくるか
                action = self.ai.get_action(state_t, evaluate)

                # 環境にて行動を実行
                self.env.execute_action(action, now_agent)

                # 環境の観測
                state_t_1, next_state_t_1, reward_t, reward_channels, game_over = self.env.observe()

                if not evaluate:
                    # 毎step終了後にtemp_Dへstore
                    self.ai.transitions.store_temp_exp(np.array((state_t)), action, reward_channels, np.array((state_t_1)), game_over)
                    self.total_training_steps += 1

                rewards.append(reward_t)
                self.score += reward_t

                # episode終了時
                if game_over == True:
                    self.temp_score += self.score

                    self.ave_score, self.temp_score = \
                        compute_ave(self.score, self.temp_score, self.ave_score, self.episode_num, div=20)

                    # 条件満たせばtemp_DをDへ保存
                    self.ai.transitions.store_exp(self.score, self.ave_score)

                    # replayのminより経験の数が多い　＋　学習フラグあり　＋　replayの頻度
                    if len(self.ai.transitions.D) >= self.replay_min_size and is_learning and \
                            self.last_episode_steps % self.ai.learning_frequency == 0:

                        # 学習を行う →　learn()　→　train_on_batch()　→　_train_on_batch()
                        self.ai.learn()

                    self.env.reset()
                    break

            # # エピソード中の最大stepまで達したら
            # if not term and self.last_episode_steps >= self.episode_max_len:
            #     print('Reaching maximum number of steps in the current episode.')
            #     term = True

        # self.fps = int(self.last_episode_steps / max(0.1, (time.time() - start_time)))
                
        return rewards

    def _reset(self):
        self.last_episode_steps = 0
        self.score = 0
        self.step_num = 0

        assert self.max_start_nullops >= self.history_len or self.max_start_nullops == 0

        # # max_start_nullops :役割が不明
        # if self.max_start_nullops != 0:
        #     num_nullops = self.rng.randint(self.history_len, self.max_start_nullops)
        #     for i in range(num_nullops - self.history_len):
        #         self.env.step(0)
        #
        # # この操作も不明
        # for i in range(self.history_len):
        #     if i > 0:
        #         self.env.step(0)
        #     obs = self.env.get_state()
        #     if obs.ndim == 1 and len(self.env.state_shape) == 2:
        #         obs = obs.reshape(self.env.state_shape)
        #     self.last_state[i] = obs

    @staticmethod
    def _update_window(window, new_value):
        window[:-1] = window[1:]
        window[-1] = new_value
        return window
                    
                    
                    
                    
                    

