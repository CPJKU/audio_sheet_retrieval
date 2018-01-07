#!/bin/bash

# example calls:
# --------------
# ./eval_models.sh cuda0 models/mutopia_ccal_cont_rsz.py ../../sheet_manager/sheet_manager/splits/all_split.yaml

# $1 ... device
# $2 ... model
# $3 ... train split

for c in mutopia_no_aug mutopia_audio_aug mutopia_sheet_aug mutopia_full_aug
do
    device="export THEANO_FLAGS=\"device=$1\""

    cmd="python run_eval.py --model $2 --data mutopia --train_split $3 --config exp_configs/$c.yaml --estimate_UV --dump_results --n_test 2000"
    echo $device
    echo $cmd
    ($device && $cmd)

    cmd="$cmd --V2_to_V1"
    echo $device
    echo $cmd
    ($device && $cmd)
done
