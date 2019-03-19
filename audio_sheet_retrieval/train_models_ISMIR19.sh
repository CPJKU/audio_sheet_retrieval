#!/bin/bash

# example calls:
# --------------
# ./train_models_ISMIR19.sh cuda0 models/mutopia_ccal_cont.py ../../sheet_manager/sheet_manager/splits/all_split.yaml

# $1 ... device
# $2 ... model
# $3 ... train split

model_names=mutopia_ccal_cont_rsz_gap mutopia_ccal_cont_rsz_dense mutopia_ccal_cont_rsz_dense_att

for c in mutopia_full_aug_sc, mutopia_full_aug_mc, mutopia_full_aug_lc
do
    device="export THEANO_FLAGS=\"device=$1\""

    cmd="python run_train.py --model $2 --data mutopia --train_split $3 --config exp_configs/$c.yaml"
    echo $device
    echo $cmd
    ($device && $cmd)

    cmd="python refine_cca.py --model $2 --data mutopia --train_split $3 --config exp_configs/$c.yaml --n_train 25000"
    echo $device
    echo $cmd
    ($device && $cmd)
done
