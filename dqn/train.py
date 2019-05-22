import os
import click
import yaml
import numpy as np

from utils import Font, set_params
from dqn.experiment import DQNExperiment
from environment.room_placement import RoomPlacement
from dqn.ai import AI

np.set_printoptions(suppress=True, linewidth=200, precision=2)
floatX = 'float32'

def worker(params):
    # 初期座標値をランダムにする
    random_state = None

    # RoomPlacement class : 室配置の環境と，その更新を定義するクラス
    env = RoomPlacement()
    params['reward_dim'] = env.reward_len

    # nb_experimentsが謎
    for ex in range(params['nb_experiments']):
        print('\n')
        print(Font.bold + Font.red + '>>>>> Experiment ', ex, ' >>>>>' + Font.end)
        print('\n')

    # AI class : 学習モデルの構築・コンパイルなどを定義するクラス
    ai = AI(env.state_shape, env.num_actions, params['action_dim'], params['reward_dim'],
                history_len=params['history_len'], gamma=params['gamma'], learning_rate=params['learning_rate'],
                epsilon=params['epsilon'], test_epsilon=params['test_epsilon'], 
                minibatch_size=params['minibatch_size'],
                replay_max_size=params['replay_max_size'], update_freq=params['update_freq'],
                learning_frequency=params['learning_frequency'], num_units=params['num_units'], rng=random_state,
                remove_features=params['remove_features'], use_mean=params['use_mean'], use_hra=params['use_hra'])

    # DQNExperiment class : 学習を試行するクラス
    expt = DQNExperiment(env=env, ai=ai, episode_max_len=params['episode_max_len'], step_max_len=params['step_max_len'],
                             history_len=params['history_len'], max_start_nullops=params['max_start_nullops'],
                             replay_min_size=params['replay_min_size'], folder_location=params['folder_location'],
                             folder_name=params['folder_name'], testing=params['test'], score_window_size=100,
                             rng=random_state)
    

    if not params['test']:
        with open(expt.folder_name + '/config.yaml', 'w') as y:
            yaml.safe_dump(params, y)  # saving params for future reference
        expt.start_training(total_eps=params['total_eps'], eps_per_epoch=params['eps_per_epoch'],
                             eps_per_test=params['eps_per_test'], is_learning=True, is_testing=True)
    else:
        raise NotImplementedError

    

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


if __name__ == '__main__':
    run(mode='hra+1')