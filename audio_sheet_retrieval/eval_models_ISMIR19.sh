#! /bin/bash

# evaluate the models obtained by running train_models_ISMIR.sh to generate Table 2 from ISMIR 2019 paper

eval_experiment () {
  cmd="python run_eval.py --model $1 --data mutopia --train_split $2 --config exp_configs/$3.yaml --dump_results --n_test 10000"
  echo $cmd
  $cmd

  cmd="$cmd --V2_to_V1"
  echo $cmd
  $cmd

  if [[ "$4" == "refine" ]]; then
    cmd="python run_eval.py --model $1 --data mutopia --train_split $2 --config exp_configs/$3.yaml --estimate_UV --dump_results --n_test 10000"
    echo $cmd
    $cmd

    cmd="$cmd --V2_to_V1"
    echo $cmd
    $cmd
  fi
}

device="export THEANO_FLAGS='device=${1}'"
echo $device
$device

# BL1
eval_experiment models/mutopia_ccal_cont_rsz_gap ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_sc refine

# BL2
eval_experiment models/mutopia_ccal_cont_rsz_dense ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_sc refine
eval_experiment models/mutopia_ccal_cont_rsz_dense ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_mc refine
eval_experiment models/mutopia_ccal_cont_rsz_dense ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_lc refine

# BL2 + Att
eval_experiment models/mutopia_ccal_cont_rsz_dense_att ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_mc refine
eval_experiment models/mutopia_ccal_cont_rsz_dense_att ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_lc refine

# BL2 + Att - CCA
eval_experiment models/mutopia_lccal_cont_rsz_dense_att.py ../../msmd/msmd/splits/all_split.yaml mutopia_full_aug_lc