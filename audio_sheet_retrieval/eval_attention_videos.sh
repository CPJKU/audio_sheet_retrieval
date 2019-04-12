#!/usr/bin/env bash

piece_names=("BeethovenLv__O79__LVB_Sonate_79_1"
             "BachJS__BWV988__bwv-988-v12"
             "BachJS__BWV817__bach-french-suite-6-menuet"
             "SchumannR__O68__schumann-op68-26-sans-titre"
             "MussorgskyM__pictures-at-an-exhibition__catacombae")

for cur_name in "${piece_names[@]}"
do
    cmd="python test_attention_video.py --model models/mutopia_ccal_cont_rsz_dense_att.py --data mutopia --train_split ../../msmd/msmd/splits/all_split.yaml"
    cmd+=" --config exp_configs/mutopia_full_aug_lc.yaml --piece $cur_name --estimate_UV"
    $cmd
done
