
from __future__ import print_function

import os
import glob
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

from audio_sheet_retrieval.config.settings import EXP_ROOT
from audio_sheet_retrieval.run_train import select_model

from collections import OrderedDict
splits = OrderedDict()
splits["bach_split_10"] = "10%"
splits["bach_split_25"] = "25%"
splits["bach_split_50"] = "50%"
splits["bach_split_75"] = "75%"
splits["bach_split"] = "100%"

if __name__ == "__main__":
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Collect evaluation results.')
    parser.add_argument('--model', help='model parameters for evaluation.', default="models/mutopia_ccal_cont_rsz.py")
    parser.add_argument('--estimate_UV', help='load re-estimated U and V.', action='store_true')
    args = parser.parse_args()

    # select model
    model, _ = select_model(args.model)

    # load model parameters
    if args.estimate_UV:
        model.EXP_NAME += "_est_UV"

    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)

    eval_template = "eval_%s_mutopia_no_aug_A2S.yaml"

    # iterate splits
    labels = []
    values = []
    for split in splits.keys():

        eval_file = eval_template % split
        eval_file = os.path.join(out_path, eval_file)

        if os.path.isfile(eval_file):

            # load results
            with open(eval_file, 'rb') as fp:
                res = yaml.load(fp)

            labels.append(splits[split])
            values.append(res["map"])

    values = np.asarray(values)
    x = np.asarray(range(len(values)))

    cols = sns.color_palette("colorblind")

    plt.figure(figsize=(3.5, 2.3))
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.22, bottom=0.23, right=0.98, top=0.95)
    plt.bar(left=x, height=values, width=0.8, color=cols[0])
    ax.yaxis.grid(True)
    plt.xticks(x + 0.4, labels)
    plt.xlabel("% of Original Train Data")
    plt.ylabel("MRR")
    plt.xlim([-0.2, 5.0])
    plt.savefig("eval_dset_size.pdf")
