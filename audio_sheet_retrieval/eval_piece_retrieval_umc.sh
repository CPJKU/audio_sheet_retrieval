#!/bin/bash

# example calls:
# --------------
# ./eval_piece_retrieval_umc.sh cuda0 models/mutopia_ccal_cont_rsz.py /home/matthias/Data/umc_chopin

# $1 ... device
# $2 ... model
# $3 ... data directory with real music

device="export THEANO_FLAGS=\"device=$1\""
echo $device

cmd="python umc_s2a_server.py --model $2 --data_dir $3 --dump_results --estimate_UV --init_audio_db --full_eval --train_split ../../sheet_manager/sheet_manager/splits/all_split.yaml --config exp_configs/mutopia_full_aug.yaml"
echo $cmd
($device && $cmd)

cmd="$cmd --real_perf"
echo $cmd
($device && $cmd)

cmd="python umc_a2s_server.py --model $2 --data_dir $3 --dump_results --estimate_UV --init_sheet_db --full_eval --train_split ../../sheet_manager/sheet_manager/splits/all_split.yaml --config exp_configs/mutopia_full_aug.yaml"
echo $cmd
($device && $cmd)

cmd="$cmd --real_perf"
echo $cmd
($device && $cmd)
