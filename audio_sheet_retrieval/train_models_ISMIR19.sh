#! /bin/sh
function run_experiment {
  device="export THEANO_FLAGS=\"device=$1\""

  cmd="python run_train.py --model $2 --data mutopia --train_split $3 --config exp_configs/$4.yaml"
  echo $device
  echo $cmd
  ($device && $cmd)

  cmd="python refine_cca.py --model $2 --data mutopia --train_split $3 --config exp_configs/$4.yaml --n_train 25000"
  echo $device
  echo $cmd
  ($device && $cmd)
}

# BL1
run_experiment cuda0 models/mutopia_ccal_cont_rsz_gap ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_sc

# BL2
run_experiment cuda0 models/mutopia_ccal_cont_rsz_dense ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_sc
run_experiment cuda0 models/mutopia_ccal_cont_rsz_dense ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_mc
run_experiment cuda0 models/mutopia_ccal_cont_rsz_dense ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_lc

# BL2 + Att
run_experiment cuda0 models/mutopia_ccal_cont_rsz_dense_att ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_mc
run_experiment cuda0 models/mutopia_ccal_cont_rsz_dense_att ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_lc
