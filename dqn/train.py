import os

import numpy as np
import yaml

from dqn.ai import AI
from dqn.experiment import DQNExperiment
from environment.room_placement import RoomPlacement
from utils import Font, set_params
import time

np.set_printoptions(suppress=True, linewidth=200, precision=2)
floatX = 'float32'

def worker(params):
    # random num generator
    np.random.seed(seed=params['random_seed'])
    random_state = np.random.RandomState(params['random_seed'])

    # RoomPlacement class : 室配置の環境と，その更新を定義するクラス
    env = RoomPlacement(params['draw_cv2_freq'], params['draw_movie_freq'], params['test_draw_cv2_freq'],
                        params['test_draw_movie_freq'], folder_name=params['folder_name'],
                        folder_location=params['folder_location'], test=params['test'])
    params['reward_dim'] = env.reward_len

    for ex in range(params['nb_experiments']):
        print('\n')
        print(Font.bold + Font.red + '>>>>> Experiment ', ex, ' >>>>>' + Font.end)
        print('\n')

    start = time.time()

    # AI class : 学習モデルの構築・コンパイルなどを定義するクラス
    ai = AI(env.state_shape, env.num_actions, params['action_dim'], params['reward_dim'],
                history_len=params['history_len'], gamma=params['gamma'], is_aggregator=params['is_aggregator'],
                learning_rate=params['learning_rate'],
                annealing=params['annealing'], annealing_episodes=params['annealing_episodes'],
                epsilon=params['epsilon'], final_epsilon=params['final_epsilon'], test_epsilon=params['test_epsilon'],
                minibatch_size=params['minibatch_size'], replay_max_size=params['replay_max_size'],
                update_freq=params['update_freq'], learning_frequency=params['learning_frequency'],
                num_units=params['num_units'], rng=random_state, test=params['test'], transfer_learn=params['transfer'],
                remove_features=params['remove_features'], use_mean=params['use_mean'], use_hra=params['use_hra'])

    print('time: AI class initialize  ' + str(round(time.time() - start, 3)))

    start = time.time()

    # DQNExperiment class : 学習を試行するクラス
    expt = DQNExperiment(env=env, ai=ai, episode_max_len=params['episode_max_len'],
                             history_len=params['history_len'], max_start_nullops=params['max_start_nullops'],
                             replay_min_size=params['replay_min_size'], folder_location=params['folder_location'],
                             folder_name=params['folder_name'], testing=params['test'], score_window_size=100,
                             rng=random_state, draw_graph_freq=params['draw_graph_freq'])

    print('time: DQN class initialize  ' + str(round(time.time() - start, 3)))

    # training
    if not params['test']:
        with open(expt.folder_name + '/config.yaml', 'w') as y:
            yaml.safe_dump(params, y)  # saving params for future reference

        start = time.time()

        expt.do_training(total_eps=params['total_eps'], eps_per_epoch=params['eps_per_epoch'],
                             eps_per_test=params['eps_per_test'], is_learning=True, is_testing=True)

        print('time: do_training  ' + str(round(time.time() - start, 3)))

    # testing
    else:
        with open(expt.folder_name + '/config.yaml', 'w') as y:
            yaml.safe_dump(params, y)  # saving params for future reference

        expt.do_testing(total_test_eps=params['total_test_eps'],
                             eps_per_test=params['eps_per_test'], is_learning=False, is_testing=True)



def run(mode):
    # modeの選択　→　hra+1
    valid_modes = ['dqn', 'dqn+1', 'hra', 'hra+1', 'all']
    assert mode in valid_modes
    if mode in ['all']:
        modes = valid_modes[:-1]
    else:
        modes = [mode]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    cfg_file = os.path.join(dir_path, 'config.yaml')
    params = yaml.safe_load(open(cfg_file, 'r'))

    for m in modes:
        params = set_params(params, m)
        worker(params)


# if __name__ == '__main__':
#     run(mode='hra+1')