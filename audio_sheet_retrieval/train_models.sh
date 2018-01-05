#!/bin/bash

# example calls:
# --------------
# ./train_models.sh cuda0 models/mutopia_ccal_cont.py ../../sheet_manager/sheet_manager/splits/all_split.yaml

# $1 ... device
# $2 ... model
# $3 ... train split

for c in mutopia_no_aug mutopia_audio_aug mutopia_sheet_aug mutopia_full_aug
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
