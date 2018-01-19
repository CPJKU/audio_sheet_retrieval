
from __future__ import print_function

import os
import glob
import yaml
import argparse
import numpy as np

from audio_sheet_retrieval.config.settings import EXP_ROOT
from audio_sheet_retrieval.run_train import select_model

from collections import OrderedDict

composers = OrderedDict()
composers["umc_mozart"] = "Mozart"
composers["umc_beethoven"] = "Beethoven"
composers["umc_chopin"] = "Chopin"


if __name__ == "__main__":
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

    eval_template = "umc_retrieval_all_split_mutopia_full_aug_%s_%s.yaml"

    # iterate composers
    for real in ["", "_real"]:
        print("\nReal: ", real)

        for composer in composers.keys():

            table_row = "%s & num_pieces" % composers[composer]

            # iterate retrieval directions
            for ret_dir in ["A2S", "S2A"]:

                aug_ranks = ["-", "-", "-", "-"]

                ret_dir += real
                eval_file = eval_template % (composer, ret_dir)
                eval_file = os.path.join(out_path, eval_file)

                if os.path.isfile(eval_file):

                    # load results
                    with open(eval_file, 'rb') as fp:
                        ranks = yaml.load(fp)
                    ranks = np.sort(ranks)
                    for idx, thr in enumerate([1, 5, 10]):
                        aug_ranks[idx] = "%d" % np.sum(ranks <= thr)
                    aug_ranks[-1] = "%d" % np.sum(ranks > thr)

                for i in range(len(aug_ranks)):
                    table_row += " & %s" % aug_ranks[i]

                # add number of candidates
                table_row = table_row.replace("num_pieces", "%d" % len(ranks))

            table_row += " \\\\"
            print(table_row)
