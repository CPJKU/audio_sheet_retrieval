"""
    Script takes a model, collects evaluation dumps for different tempi and visualizes those.
"""
import os
import argparse
import glob
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main(path_exp, direction):
    results = []

    for cur_path_yaml in glob.glob(os.path.join(path_exp, '*.yaml')):
        # check for query direction
        if direction not in cur_path_yaml:
            continue

        cur_tempo = int(os.path.splitext(os.path.basename(cur_path_yaml))[0].split('_')[-1]) / 1000

        with open(cur_path_yaml, 'rb') as hdl:
            cur_results = yaml.load(hdl)

        cur_results_dict = dict()
        cur_results_dict['tempo'] = cur_tempo
        cur_results_dict['map'] = cur_results['map']
        cur_results_dict['med_rank'] = cur_results['med_rank']
        cur_results_dict['recall_at_1'] = cur_results['recall_at_k']['1']
        cur_results_dict['recall_at_5'] = cur_results['recall_at_k']['5']
        cur_results_dict['recall_at_10'] = cur_results['recall_at_k']['10']
        cur_results_dict['recall_at_25'] = cur_results['recall_at_k']['25']
        results.append(cur_results_dict)

    results = pd.DataFrame(results).sort_values(by=['tempo', ])

    # map over tempo
    plt.figure()
    x_ticks = range(results.shape[0])
    plt.bar(x=x_ticks, height=results['map'])
    plt.ylim([0, 1])
    plt.xlabel('Tempo Ratio')
    plt.ylabel('MRR')
    plt.xticks(x_ticks, results['tempo'])

    # recall at k over tempo
    plt.figure()
    n_recalls = 4
    bar_width = 0.1

    for cur_tempo_idx in range(results.shape[0]):
        x_positions = cur_tempo_idx + np.linspace(- 2 * bar_width, 2 * bar_width, n_recalls)
        heights = results.loc[cur_tempo_idx, :]
        heights = heights[['recall_at_1', 'recall_at_5', 'recall_at_10', 'recall_at_25']]
        plt.bar(x=x_positions, height=heights, width=0.1)

    plt.xlabel('Tempo Ratio')
    plt.ylabel('R@k')
    plt.ylim([0, 100])
    plt.xticks(x_ticks, results['tempo'])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize MAP for different tempi.')
    parser.add_argument('--exp', help='path to experiments folder.', default="")
    parser.add_argument('--dir', help='query direction.', default="A2S")
    args = parser.parse_args()

    main(args.exp, args.dir)
