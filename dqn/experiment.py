import time
import numpy as np
from utils import Font, plot_and_write, create_folder
import copy

class DQNExperiment(object):
    def __init__(self, env, ai, episode_max_len, step_max_len, history_len=1, max_start_nullops=1, replay_min_size=0,
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
        self.step_max_len = step_max_len
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
                #
                self.score_agent_window = self._update_window(self.score_agent_window, self.score_agent)
                self.steps_agent_window = self._update_window(self.steps_agent_window, self.last_episode_steps)
            else:
                # episode_numを足す
                self.episode_num += 1
        return np.mean(scores), np.mean(steps)

    def _do_episode(self, is_learning=True):
        rewards = []
        self.env.reset()
        term = False
        start_time = time.time()
        
        # eps中において，termに達するまでstepを行う
        while not term:
            reward, term = self._step(evaluate=not is_learning)
            rewards.append(reward)

            # replayのminより経験の数が多い　＋　学習フラグあり　＋　replayの頻度
            if self.ai.transitions.size >= self.replay_min_size and is_learning and \
                    self.last_episode_steps % self.ai.learning_frequency == 0:

                # 学習を行う →　learn()　→　train_on_batch()　→　_train_on_batch()
                self.ai.learn()

            self.score_agent += reward

            # エピソード中の最大stepまで達したら
            if not term and self.last_episode_steps >= self.episode_max_len:
                print('Reaching maximum number of steps in the current episode.')
                term = True
                
        return rewards

        
    def _step(self, evaluate=False):
        # stepを足す
        self.last_episode_steps += 1
        
        action = self.ai.get_action(self.last_state, evaluate)

        # env classのstepから、tにおける情報を受ける
        new_obs, reward, game_over, info = self.env.step(action, now_agetn, step=self.last_episode_steps)
        
        
    def start_training(self, total_eps=5000, eps_per_epoch=100, eps_per_test=100, is_learning=True, is_testing=False):
        # グラフ描画用のリスト
        n_episode_list = []
        ave_reward_list = []

        # total eps に達するまでepsを行う
        while self.episode_num < self.episode_max_len:
            self.episode_num += 1
            print(Font.yellow + Font.bold + 'Training ... ' + str(self.episode_num) + '/' + str(self.episode_max_len) + Font.end,
                  end='\n')

            cnt_step = 0
            cnt_reward = 0

            loss = 0
            Q_max = 0

            steps =[]
            rewards = []

            # 報酬の最大と平均を出力する変数
            max_reward = 0
            ave_reward = 0
            temp_reward = 0
            temp_ave_reward = 0

            self.env.reset()
            start_time = time.time()

            # 環境の観測．今のエージェントのstate_t+1, 次のエージェントのstate_t+1
            state_t_1, next_state_t_1, reward_t, reward_t_chan, terminal = self.env.observe()

            # eps中において，termに達するまでstepを行う
            while not terminal:
                for now_agent in range(self.num_agents):
                    # 前のエージェントのstate_t+1で今のstate_tを初期化
                    state_t = copy.deepcopy(next_state_t_1)

                    # 環境にて行動を選択
                    action_t = self.ai.get_action(state_t, evaluate=not is_learning)

                    # 環境にて行動を実行
                    self.env.execute_action(action_t, now_agent, cnt_step)

                    # 環境の観測
                    state_t_1, next_state_t_1, reward_t, reward_t_chan, terminal = self.env.observe()

                    # 経験を一時蓄積する
                    self.ai.store_temp_exp(state_t, action_t, reward_t_chan, state_t_1, terminal)

                    cnt_step += 1
                    cnt_reward += reward_t

                    if terminal:
                        print('cnt_reward: ' + str(cnt_reward))
                        tart_replay = False

                        # 平均値出力用のtemp
                        temp_reward += cnt_reward

                        # rewardの平均値を出力　→　最大ならば更新
                        div = 20
                        if self.episode_num % div == 0:
                            temp_ave_reward = temp_reward / div
                            if temp_reward / div > ave_reward:
                                ave_reward = temp_reward / div
                                print("average_reward: " + str(ave_reward))
                            # グラフ出力用の変数
                            n_episode_list.append(self.episode_num)
                            ave_reward_list.append(temp_ave_reward)
                            temp_reward = 0

                        # rewardの最大値を出力
                        if cnt_reward > max_reward:
                            max_reward = cnt_reward
                            print("max_reward: " + str(max_reward))

                        # store，Dへの蓄積数が一定数を超えるとフラグを返す
                        start_replay = self.ai.store_exp(cnt_reward, ave_reward)

                        # start_replayフラグなら
                        if start_replay:
                            # target_update
                            if self.episode_num % self.ai.update_freq == 0:
                                # agent.end_session()
                                self.ai.update_weights([])
                                print('update_target_mode')

                            # 探索の閾値を更新
                            self.ai.update_exploration()

                            # exp_replay
                            # agent.experience_replay(episode)
                            print('exp_replay')

                            # 複数回exp_replay
                            for replay_num in range(10):
                                # print('---: ' + str(replay_num) + 'exp_replay---')
                                self.ai.experience_replay()
                                if replay_num == 9:
                                    break

                        break
                    
                    
                    
                    
                    
                    

