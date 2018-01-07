#!/bin/bash

# example calls:
# --------------
# ./eval_piece_retrieval.sh cuda0 models/mutopia_ccal_cont_rsz.py ../../sheet_manager/sheet_manager/splits/all_split.yaml

# $1 ... device
# $2 ... model
# $3 ... train split

for c in mutopia_no_aug mutopia_full_aug
do
    device="export THEANO_FLAGS=\"device=$1\""

    cmd="python audio_sheet_server.py --model $2 --train_split $3 --config exp_configs/$c.yaml --full_eval --init_sheet_db --estimate_UV --dump_results"
    echo $device
    echo $cmd
    ($device && $cmd)

    cmd="python sheet_audio_server.py --model $2 --train_split $3 --config exp_configs/$c.yaml --full_eval --init_audio_db --estimate_UV --dump_results"
    echo $device
    echo $cmd
    ($device && $cmd)
done
