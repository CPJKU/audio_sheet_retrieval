#!/bin/bash

# example calls:
# --------------
# ./train_models_dset_size.sh cuda0 models/mutopia_ccal_cont_rsz.py

# $1 ... device
# $2 ... model

for s in bach_split_10 bach_split_25 bach_split_50 bach_split_75
do
    device="export THEANO_FLAGS=\"device=$1\""

#    cmd="python run_train.py --model $2 --data mutopia --train_split ../../sheet_manager/sheet_manager/splits/$s.yaml --config exp_configs/mutopia_no_aug.yaml"
#    echo $device
#    echo $cmd
#    ($device && $cmd)

    cmd="python refine_cca.py --model $2 --data mutopia --train_split ../../sheet_manager/sheet_manager/splits/$s.yaml --config exp_configs/mutopia_no_aug.yaml --n_train 25000"
    echo $device
    echo $cmd
    ($device && $cmd)
done
