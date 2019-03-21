#!/bin/bash

# example calls:
# --------------
# ./eval_models_tempo.sh cuda0 models/mutopia_ccal_cont_rsz_dense_att.py ../../msmd/msmd/splits/all_split.yaml exp_configs/mutopia_full_aug_lc.yaml

# $1 ... device
# $2 ... model
# $3 ... train split
# $4 ... config file

for t in 0.5 0.75 1.0 1.25 1.5 1.75 2.0
do
    device="export THEANO_FLAGS=\"device=$1\""

    cmd="python run_eval.py --model $2 --data mutopia --train_split $3 --config $4 --estimate_UV --dump_results --test_tempo $t --n_test 10000"
    echo $device
    echo $cmd
    ($device && $cmd)

    cmd="$cmd --V2_to_V1"
    echo $device
    echo $cmd
    ($device && $cmd)
done
