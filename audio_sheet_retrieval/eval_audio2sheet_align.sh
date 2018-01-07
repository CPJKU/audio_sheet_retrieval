#!/bin/bash

# example calls:
# --------------
# ./eval_audio2sheet_align.sh cuda0 models/mutopia_ccal_cont_rsz.py ../../sheet_manager/sheet_manager/splits/all_split.yaml

# $1 ... device
# $2 ... model
# $3 ... train split

for m in baseline pydtw
do
    device="export THEANO_FLAGS=\"device=$1\""

    cmd="python audio2sheet_align.py --model $2 --align_by $m --train_split $3 --config exp_configs/mutopia_full_aug.yaml --estimate_UV --dump_alignment --dump_alignment"
    echo $device
    echo $cmd
    ($device && $cmd)
done
