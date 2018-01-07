
from __future__ import print_function

import os
import glob
import yaml
import argparse
import numpy as np

from audio_sheet_retrieval.config.settings import EXP_ROOT
from audio_sheet_retrieval.run_train import select_model

from collections import OrderedDict

aug_mapping = OrderedDict()
aug_mapping["mutopia_no_aug"] = "none"
aug_mapping["mutopia_sheet_aug"] = "sheet"
aug_mapping["mutopia_audio_aug"] = "audio"
aug_mapping["mutopia_full_aug"] = "full"


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

    eval_template = "eval_%s_%s_%s.yaml"

    # iterate retrieval directions
    for ret_dir in ["A2S", "S2A"]:

        print("\nRetrieval Direction:", ret_dir)

        # iterate augmentation strategies
        for aug in aug_mapping.keys():

            table_row = "%s " % aug_mapping[aug]

            # iterate splits
            for split in ["bach_split", "bach_out_split", "all_split"]:

                eval_file = eval_template % (split, aug, ret_dir)
                eval_file = os.path.join(out_path, eval_file)

                if os.path.isfile(eval_file):

                    # load results
                    with open(eval_file, 'rb') as fp:
                        res = yaml.load(fp)

                    table_row += " & %.2f & %.2f & %.2f & %d" % (res["recall_at_k"]["1"] / 100, res["recall_at_k"]["25"] / 100, res["map"], res["med_rank"])

                else:
                    table_row += " & - & - & - & -"

            table_row += " \\\\"

            print(table_row)