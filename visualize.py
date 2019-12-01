import csv
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import numpy as np
import copy

# color_schemes = {'scores': 'dodgerblue', 'scores_connect' : 'dodgerblue', 'scores_shape' : 'yellowgreen', 'scores_area' : 'tomato', 'scores_collision' : 'khaki'}
color_schemes = {'scores': 'darkorange', 'scores_connect' : 'dodgerblue', 'scores_shape' : 'dimgrey', 'scores_area' : 'dimgrey', 'scores_collision' : 'dimgrey'}
x_label = 'Episode'
y_label = 'Score'


def set_seaborn():
    sns.set()
    # sns.set_style("whitegrid", {'grid.linestyle': '--'})
    sns.set_context("paper", 1.5, {"lines.linewidth": 1})
    # sns.set_palette("winter_r", 8, 1)

def draw_line_graph_from_list(file_name, save_name):
    with open(file_name) as f:
        # csv読み込み　headerのみ取り出し
        reader = csv.reader(f)
        csv_list = [row for row in reader]
        header = copy.copy(csv_list[0])
        del csv_list[0]

        # csv_listをfloatに変換
        csv_list_float = [[float(v) for v in row] for row in csv_list]

        # 転置行列でグラフ描画用に成形
        csv_list_T = (np.array(csv_list_float).T).tolist()

    for num, str in enumerate(header):
        if str == '' or str == 'episode':
            pass
        else:
            if str == 'scores':
                plt.plot(csv_list_T[1], csv_list_T[num], alpha=0.2, linewidth=1.0, color=color_schemes[str])
                plt.legend()

                # 以下関数化
                scores_all = csv_list_T[header.index(str)]
                scores_ave = []
                scores_ave_eps = []
                n_ave = 10
                for i in range(len(scores_all) // n_ave):
                    # scores_ave.extend(sum(scores_all[i*n_ave:i*n_ave+n_ave]))
                    score_ave = sum(scores_all[i * n_ave:i * n_ave + n_ave]) / n_ave
                    scores_ave.append(score_ave)
                    scores_ave_eps.append(i * n_ave * 10)
                plt.plot(scores_ave_eps, scores_ave, linewidth=2.0, color=color_schemes[str])

            else:
                plt.plot(csv_list_T[1], csv_list_T[num], alpha=0.2, linewidth=1.0, color=color_schemes[str])

                # 以下関数化
                scores_all = csv_list_T[header.index(str)]
                scores_ave = []
                scores_ave_eps = []
                n_ave = 10
                for i in range(len(scores_all) // n_ave):
                    # scores_ave.extend(sum(scores_all[i*n_ave:i*n_ave+n_ave]))
                    score_ave = sum(scores_all[i * n_ave:i * n_ave + n_ave]) / n_ave
                    scores_ave.append(score_ave)
                    scores_ave_eps.append(i * n_ave * 10)
                plt.plot(scores_ave_eps, scores_ave, linewidth=1.5, color=color_schemes[str])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_name)
    plt.close()


def draw_line_graph_from_df(file_name, save_name, x_axis, y_axes=None, y_axes_bold=None):
    df = pd.read_csv(file_name)
    # print(df['scores'])
    # print(df.head())
    filter = [x_axis] + y_axes
    filter_bold = [x_axis] + ['scores']
    title = os.path.basename(file_name)
    df[filter].plot(x=x_axis, title=title, grid=True, alpha=0.3)
    plt.savefig(save_name)
    plt.close()


set_seaborn()
draw_line_graph_from_list('scores_4-9.csv', '4-9.png')