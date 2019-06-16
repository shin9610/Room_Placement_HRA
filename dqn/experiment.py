import copy
import time

import numpy as np

from utils import Font, plot_and_write, create_folder, compute_ave


class DQNExperiment(object):
    def __init__(self, env, ai, episode_max_len, history_len=1, max_start_nullops=1, replay_min_size=0,
                 score_window_size=100, rng=None, draw_graph_freq=100, folder_location='/experiments/',
                 folder_name='expt', testing=False):
        self.rng = rng
        self.fps = 0
        self.episode_num = 1
        self.last_episode_steps = 0
        self.total_training_steps = 0
        self.score = 0
        self.temp_scores = 0
        self.ave_score = 0
        self.score_channel = np.zeros(env.reward_len)
        self.eval_steps = []
        self.eval_scores = []
        self.eval_scores_connect = []
        self.eval_scores_shape = []
        self.eval_scores_area = []
        self.elapsed_times = []
        self.total_time = 0
        self.learn_time = 0
        self.learn_times = []
        self.env = env
        self.ai = ai
        self.history_len = history_len
        self.max_start_nullops = max_start_nullops

        self.folder_name, self.folder_name_images, self.folder_name_movies = \
            create_folder(folder_location, folder_name, test=testing)

        self.episode_max_len = episode_max_len
        self.num_agents = env.n_agents
        self.score_window = np.zeros(score_window_size)
        self.steps_agent_window = np.zeros(score_window_size)
        self.replay_min_size = max(self.ai.minibatch_size, replay_min_size)
        self.last_state = np.empty(tuple([self.history_len] + self.env.state_shape), dtype=np.uint8)

        self.draw_graph_freq = draw_graph_freq

    def do_testing(self, total_test_eps=5, eps_per_test=1, is_learning=False, is_testing=True):
        print(Font.cyan + Font.bold + 'Testing ... '  + Font.end, end='\n')
        for i in range(total_test_eps):
            test_scores, test_scores_connect, test_scores_shape, test_scores_area, _, _ = \
                self.do_episodes(number=eps_per_test, is_learning=False)
            print('Test_Score: ' + str(test_scores) + '\n')
            print('connect/shape/area: ' + str(test_scores_connect) + '/' + str(test_scores_shape) + '/' + str(test_scores_area))
            # 動画の作成
            self.env.movie(self.episode_num, self.folder_name_images, self.folder_name_movies)

    def do_training(self, total_eps=5000, eps_per_epoch=10, eps_per_test=100, is_learning=True, is_testing=True):
        # total eps に達するまでepsを行う
        while self.episode_num < total_eps:
            print(Font.yellow + Font.bold + 'Training ... ' + str(self.episode_num) + '/' + str(total_eps) + Font.end,
                  end='\n')
            _, _, _, _, elapsed_times, learn_times = self.do_episodes(number=eps_per_epoch, is_learning=is_learning)
            # self.do_episodes(number=eps_per_epoch, is_learning=is_learning)
            # graph(self.episode_num, scores, self.draw_graph_freq)

            if is_testing:
                print('testing')
                print('epsilon: ' + str(self.ai.epsilon))
                eval_scores, eval_scores_connect, eval_scores_shape, eval_scores_area, _, _ = \
                    self.do_episodes(number=eps_per_test, is_learning=False)

                self.eval_scores.append(eval_scores)
                self.eval_scores_connect.append(eval_scores_connect)
                self.eval_scores_shape.append(eval_scores_shape)
                self.eval_scores_area.append(eval_scores_area)

                self.elapsed_times.append(elapsed_times)
                self.learn_times.append(learn_times)

                print('Eval_Score: ' + str(eval_scores) + '\n' + 'Time: ' + str(elapsed_times) + '\n' +
                      'Learn_Time: ' + str(learn_times) + '\n' + 'Total_time: ' + str(self.total_time))

                # 動画の作成
                self.env.movie(self.episode_num, self.folder_name_images, self.folder_name_movies)

                # グラフの作成
                # plot_and_write(plot_dict={'scores': self.eval_scores}, loc=self.folder_name + "/scores",
                #                x_label="Episodes", y_label="Scores", title="", kind='line', legend=True)

                plot_and_write(plot_dict={'scores': self.eval_scores, 'scores_connect': self.eval_scores_connect,
                                          'scores_shape': self.eval_scores_shape, 'scores_area': self.eval_scores_area},
                               loc=self.folder_name + "/scores",
                               x_label="Episodes", y_label="Scores", title="", kind='line', legend=True)

                plot_and_write(plot_dict={'times': self.elapsed_times}, loc=self.folder_name + "/times",
                               x_label="Episodes", y_label="Times", title="", kind='line', legend=True)
                plot_and_write(plot_dict={'learn_times': self.learn_times}, loc=self.folder_name + "/learn_times",
                               x_label="Episodes", y_label="Times", title="", kind='line', legend=True)

                self.ai.dump_network(weights_file_path=self.folder_name + '/q_network_weights.h5',
                                     overwrite=True)

    def do_episodes(self, number, is_learning=True):
        times = []
        learn_times = []
        scores = []
        scores_connect = []
        scores_shape = []
        scores_area = []
        steps = []

        for num in range(number):
            start_time = time.time()

            self._do_episode(is_learning=is_learning, evaluate=not is_learning)
            scores.append(self.score)
            scores_connect.append(self.score_channel[0])
            scores_shape.append(self.score_channel[1])
            scores_area.append(self.score_channel[2])

            steps.append(self.last_episode_steps)
            learn_times.append(self.learn_time)
            # print(self.learn_time)

            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            self.total_time += elapsed_time
            # print(elapsed_time)

            if not is_learning:
                self.score_window = self._update_window(self.score_window, self.score)
                # self.steps_agent_window = self._update_window(self.steps_agent_window, self.last_episode_steps)
            else:
                # episode_numを足す
                self.episode_num += 1

        return np.mean(scores), np.mean(scores_connect), np.mean(scores_shape), np.mean(scores_area), \
               np.mean(times), np.mean(learn_times)

    def _do_episode(self, is_learning=True, evaluate=False):
        rewards = []

        self.env.reset()
        self._reset()

        # 環境の観測
        state_t_1, next_state_t_1, reward_t, reward_channels, game_over = self.env.observe()

        # eps中において，termに達するまでstepを行う
        while not game_over:
            # 各agent順に行動
            for now_agent in range(self.env.n_agents):
                self.last_episode_steps += 1
                
                state_t = copy.deepcopy(next_state_t_1)

                # 行動前の報酬の観測
                _, pre_reward_channels = self.env.reward_condition(now_agent)

                # 行動選択　→　last_stateをどこから持ってくるか
                action = self.ai.get_action(state_t, evaluate, pre_reward_channels)

                # 環境にて行動を実行
                self.env.execute_action(action, now_agent)

                # 行動後の環境の観測
                # state_t_1, next_state_t_1, reward_t, reward_channels, game_over = self.env.observe()
                state_t_1, next_state_t_1, reward_t, reward_channels, game_over = self.env.observe()

                rewards.append(reward_t)
                self.score += reward_t
                self.score_channel += np.array(reward_channels)

                # replayのminより経験の数が多い　＋　学習フラグあり　＋　replayの頻度
                if len(self.ai.transitions.D) >= self.replay_min_size and is_learning and \
                        self.last_episode_steps % self.ai.learning_frequency == 0:

                    # 学習を行う →　learn()　→　train_on_batch()　→　_train_on_batch()
                    _, self.learn_time = self.ai.learn()

                # temp_Dを保存　→ term = Falseで保存
                if not evaluate:
                    self.ai.transitions.store_temp_exp(np.array((state_t)), action, reward_channels, np.array((state_t_1)), False)
                    self.total_training_steps += 1

                # フレーム画像の描画
                self.env.draw_cv2(now_agent, action, self.last_episode_steps, reward_channels,
                                  self.score, self.episode_num, self.folder_name_images)

                # episode終了時
                if game_over == True:
                    # 条件満たせばtemp_DをDへ保存
                    if not evaluate:
                        self.temp_scores += self.score
                        self.ave_score, self.temp_scores = \
                            compute_ave(self.score, self.temp_scores, self.ave_score, self.episode_num, div=20)
                        self.ai.transitions.store_exp(self.score, self.ave_score)
                        self.ai.update_epsilon()
                        # print('store_temp_D')

                    self.env.reset()
                    break


            # # エピソード中の最大stepまで達したら
            # if not term and self.last_episode_steps >= self.episode_max_len:
            #     print('Reaching maximum number of steps in the current episode.')
            #     term = True

        # self.fps = int(self.last_episode_steps / max(0.1, (time.time() - start_time)))

        print(self.score)
        return rewards

    def _reset(self):
        self.last_episode_steps = 0
        self.score = 0
        self.score_channel = np.zeros(self.env.reward_len)
        self.learn_time = 0

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
                    
                    
                    
                    
                    

