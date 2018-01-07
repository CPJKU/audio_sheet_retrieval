
from __future__ import print_function

import os
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from audio_sheet_retrieval.config.settings import EXP_ROOT
from audio_sheet_retrieval.run_train import select_model

sns.set_style("ticks")

method_dict = dict()
method_dict["baseline"] = "Baseline"
method_dict["pydtw"] = "Embedding DTW"

split_mapping = dict()
split_mapping["bach_split"] = "bach-set"
split_mapping["bach_out_split"] = "bach-out"
split_mapping["all_split"] = "all"

if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Collect evaluation results.')
    parser.add_argument('--model', help='model parameters for evaluation.', default="models/mutopia_ccal_cont_rsz.py")
    parser.add_argument('--estimate_UV', help='load re-estimated U and V.', action='store_true')
    args = parser.parse_args()

    # select model
    model, _ = select_model(args.model)
    if args.estimate_UV:
        model.EXP_NAME += "_est_UV"
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)

    eval_template = "alignment_res_%s_mutopia_full_aug_%s.pkl"

    # init result data structure
    summary = pd.DataFrame(columns=['align_by', 'split', '|Alignment Error|'])

    # iterate splits
    for split in ["bach_split", "bach_out_split", "all_split"]:

        for align_by in ["baseline", "pydtw"]:

            eval_file = eval_template % (split, align_by)
            eval_file = os.path.join(out_path, eval_file)

            # load results
            with open(eval_file, "rb") as fp:
                piece_pxl_errors = pickle.load(fp)

            # collect errors of all pieces
            all_errors = np.zeros((0, ), dtype=np.float32)
            for piece in piece_pxl_errors.keys():
                all_errors = np.concatenate((all_errors, piece_pxl_errors[piece]))

            # compute normalized abs errors
            abs_all_errors = np.abs(all_errors)
            abs_all_errors /= 835
            abs_all_errors *= 100

            df = pd.DataFrame()
            df['Split'] = [split_mapping[split]] * len(abs_all_errors)
            df['Aligned by'] = [method_dict[align_by]] * len(abs_all_errors)
            df['Normalized |Alignment Error|'] = abs_all_errors

            summary = summary.append(df)

            print(split, align_by, "%.2f" % np.median(abs_all_errors))

    # create boxplot
    plt.figure(figsize=(4.5, 4.5))
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.98, top=0.98)
    sns.boxplot(x="Split", y="Normalized |Alignment Error|", hue="Aligned by", data=summary,
                palette="colorblind", showfliers=False)
    sns.despine(offset=10, trim=True)
    plt.ylim([0, 55])
    # plt.grid("on")
    ax.yaxis.grid(True)
    plt.savefig("alignment_boxplot.pdf")

    # per_method_results = dict()
    #
    # for res_file in res_files:
    #     print ""
    #     print res_file
    #
    #     # load results
    #     with open(res_file, "rb") as fp:
    #         piece_pxl_errors = pickle.load(fp)
    #
    #     # collect errors
    #     all_pxl_errors = []
    #     for piece, errs in piece_pxl_errors.iteritems():
    #         all_pxl_errors.extend(errs)
    #
    #     all_pxl_errors = np.array(all_pxl_errors)
    #     abs_pxl_errors = np.abs(all_pxl_errors)
    #     abs_pxl_errors /= 835
    #
    #     # report results for piece
    #     print "Mean Error:   %.3f" % np.mean(abs_pxl_errors)
    #     print "Median Error: %.3f" % np.median(abs_pxl_errors)
    #     print "Std Error:    %.3f" % np.std(abs_pxl_errors)
    #
    #     # keep per piece errors
    #     method = res_file.rsplit('_')[-1].split('.pkl')[0]
    #     per_method_results[method] = abs_pxl_errors
    #     # per_method_results[method] = all_pxl_errors
    #
    # # initialize data frame
    # methods = []
    # abs_pxl_errors = []
    # for k, v in per_method_results.iteritems():
    #     methods += [k] * len(v)
    #     abs_pxl_errors.append(v)
    #
    # abs_pxl_errors = np.asarray(abs_pxl_errors).T
    #
    # # fix labels
    # methods = np.asarray(methods)
    # methods[methods == 'baseline'] = 'Linear'
    # methods[methods == 'pydtw'] = 'DTW'
    #
    # df = pd.DataFrame()
    # df['Alignment Method'] = methods
    # df['|Alignment Error|'] = np.concatenate([abs_pxl_errors[:, i] for i in xrange(n_methods)])
    # # figures
    # sns.set(style="ticks", palette="muted", color_codes=True)
    #
    # plt.figure(figsize=(7, 3))
    # plt.clf()
    #
    # ax = sns.boxplot(x="|Alignment Error|", y="Alignment Method", data=df,
    #                  whis=np.inf, color="b")
    #
    # ax.xaxis.label.set_fontsize(16)
    # ax.yaxis.label.set_fontsize(16)
    #
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    #
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    #     tick.label.set_rotation(45)
    #
    # plt.subplots_adjust(left=0.18, bottom=0.25)
    #
    # plt.show(block=True)
