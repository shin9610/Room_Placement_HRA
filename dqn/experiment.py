import time
import numpy as np
from utils import Font, plot_and_write, create_folder
import copy

class DQNExperiment(object):
    def __init__(self, env, ai, episode_max_len, history_len=1, max_start_nullops=1, replay_min_size=0,
                 score_window_size=100, rng=None, folder_location='/experiments/', folder_name='expt', testing=False):
        self.rng = rng
        self.fps = 0
        self.episode_num = 0
        self.last_episode_steps = 0
        self.total_training_steps = 0
        self.score_agent = 0
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
        self.score_agent_window = np.zeros(score_window_size)
        self.steps_agent_window = np.zeros(score_window_size)
        self.replay_min_size = max(self.ai.minibatch_size, replay_min_size)
        self.last_state = np.empty(tuple([self.history_len] + self.env.state_shape), dtype=np.uint8)


    def do_training(self, total_eps=5000, eps_per_epoch=100, eps_per_test=100, is_learning=True, is_testing=True):
        # total eps に達するまでepsを行う
        while self.episode_num < total_eps:
            print(Font.yellow + Font.bold + 'Training ... ' + str(self.episode_num) + '/' + str(total_eps) + Font.end,
                  end='\n')
            self.do_episodes(number=eps_per_epoch, is_learning=is_learning)

            if is_testing:
                eval_scores, eval_steps = self.do_episodes(number=eps_per_test, is_learning=False)
                self.eval_steps.append(eval_steps)
                self.eval_scores.append(eval_scores)
                plot_and_write(plot_dict={'steps': self.eval_steps}, loc=self.folder_name + "/steps",
                               x_label="Episodes", y_label="Steps", title="", kind='line', legend=True)
                plot_and_write(plot_dict={'scores': self.eval_scores}, loc=self.folder_name + "/scores",
                               x_label="Episodes", y_label="Scores", title="", kind='line', legend=True)
                self.ai.dump_network(weights_file_path=self.folder_name + '/q_network_weights.h5',
                                     overwrite=True)

    def do_episodes(self, number=1, is_learning=True):
        scores = []
        steps = []
        
        # eps中において，eps per epochに達するまでepsを行う　→　1回に指定
        for num in range(number):
            self._do_episode(is_learning=is_learning)
            scores.append(self.score_agent)
            steps.append(self.last_episode_steps)

            # 学習が終了したなら
            if not is_learning:
                # 元コードの処理が不明　→　passで
                # self.score_agent_window = self._update_window(self.score_agent_window, self.score_agent)
                # self.steps_agent_window = self._update_window(self.steps_agent_window, self.last_episode_steps)
                pass
            else:
                # episode_numを足す
                self.episode_num += 1
        return np.mean(scores), np.mean(steps)

    def _do_episode(self, is_learning=True):
        rewards = []

        self.env.reset()
        self._reset()
        term = False

        start_time = time.time()
        
        # eps中において，termに達するまでstepを行う
        while not term:
            # 各agent順に行動
            for now_agent in range(self.env.n_agents):
                reward, term = self._step(now_agent, evaluate=not is_learning)
                rewards.append(reward)
                self.score_agent += reward

            if term == True:
                # replayのminより経験の数が多い　＋　学習フラグあり　＋　replayの頻度
                if self.ai.transitions.size >= self.replay_min_size and is_learning and \
                        self.last_episode_steps % self.ai.learning_frequency == 0:

                    # 学習を行う →　learn()　→　train_on_batch()　→　_train_on_batch()
                    self.ai.learn()

            # # エピソード中の最大stepまで達したら
            # if not term and self.last_episode_steps >= self.episode_max_len:
            #     print('Reaching maximum number of steps in the current episode.')
            #     term = True

        # self.fps = int(self.last_episode_steps / max(0.1, (time.time() - start_time)))
                
        return rewards

    def _reset(self):
        self.last_episode_steps = 0
        self.score_agent = 0

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

    def _step(self, now_agent, evaluate=False):
        self.last_episode_steps += 1

        # 初めのステップの時の初期化
        if self.last_episode_steps == 1:
            state_t_1, next_state_t_1, reward_t, reward_channels, game_over = self.env.observe()

        state_t = copy.deepcopy(next_state_t_1)

        # 行動選択　→　last_stateをどこから持ってくるか
        action = self.ai.get_action(state_t, evaluate)

        # 環境にて行動を実行
        self.env.execute_action(action, now_agent)

        # new_obs, reward, game_over, info = self.env.observe
        state_t_1, next_state_t_1, reward_t, reward_channels, game_over = self.env.observe()

        if not evaluate:
            # store memory　→　Dで実装しなおしてもよい
            self.ai.transitions.add(s=self.last_state[-1].astype('float32'), a=action, r=reward_channels, t=game_over)
            self.total_training_steps += 1

        return reward_t, game_over

    @staticmethod
    def _update_window(window, new_value):
        window[:-1] = window[1:]
        window[-1] = new_value
        return window
                    
                    
                    
                    
                    

