#! /bin/sh
run_experiment () {
  cmd="python run_train.py --model $1 --data mutopia --train_split $2 --config exp_configs/$3.yaml"
  echo $cmd
  $cmd

  cmd="python refine_cca.py --model $1 --data mutopia --train_split $2 --config exp_configs/$3.yaml --n_train 25000"
  echo $cmd
  $cmd
}


device="export THEANO_FLAGS='device=${1}'"
echo $device
$device

# BL1
run_experiment models/mutopia_ccal_cont_rsz_gap ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_sc

# BL2
run_experiment models/mutopia_ccal_cont_rsz_dense ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_sc
run_experiment models/mutopia_ccal_cont_rsz_dense ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_mc
run_experiment models/mutopia_ccal_cont_rsz_dense ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_lc

# BL2 + Att
run_experiment models/mutopia_ccal_cont_rsz_dense_att ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_mc
run_experiment models/mutopia_ccal_cont_rsz_dense_att ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_lc
