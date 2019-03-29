#!/bin/bash

# example calls:
# --------------
# ./eval_models_tempo.sh cuda0 models/mutopia_ccal_cont_rsz_dense_att.py ../../msmd/msmd/splits/all_split.yaml exp_configs/mutopia_full_aug_lc.yaml

# $1 ... device
# $2 ... model
# $3 ... train split
# $4 ... config file

device="export THEANO_FLAGS=\"device=$1\""

for t in 0.5 0.75 1.0 1.25 1.5 1.75 2.0
do
    for est_UV in "" "--estimate_UV"
    do
        # sheet to audio
        cmd="python run_eval.py --model $2 --data mutopia --train_split $3 --config $4 --dump_results --test_tempo $t --n_test 10000 $est_UV"
        echo $device
        echo $cmd
        ($device && $cmd)

        # audio to sheet
        cmd="$cmd --V2_to_V1"
        echo $device
        echo $cmd
        ($device && $cmd)
    done
done
