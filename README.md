
End-to-End Audio – Sheet Music Correspondence Learning and Retrieval
====================================================================
This repository contains all experimental code for reproducing the results
reported in our manuscript:

>End-to-End Audio – Sheet Music Correspondence Learning and Retrieval.<br>
Anonymous Authors.<br>
Under review for TISMIR Volume 1, 2018

The retrieval methodology employed is this work is based on
the *CCA Projection Layer* especially designed for the task
of cross-modality retrieval.
An implementation of this layer can be found in this sub folder of this repository:
*audio_sheet_retrieval/models/lasagne_extensions/layers/cca.py*.
And a detailed description is provided in:

>End-to-End Cross-Modality Retrieval with CCA Projections and Pairwise Ranking Loss.<br>
M Dorfer, J Schlüter, A Vall, F Korzeniowski, G Widmer.<br>
arXiv preprint arXiv:1705.06979


Requirements
------------
- python-opencv
- pyyaml
- numpy
- matplotlib
- theano
- lasagne
- tqdm

Preparation
-----------
Before you can start running the code make sure that all paths are configured correctly.
In particular, you have to specify two paths in the file *config/settings.py*:
```
# path where model folder gets created and parameters and results get dumped
EXP_ROOT = "/home/matthias/experiments/audio_sheet_retrieval/"
# path where to find the data (In our case the root directory of the msmd dataset)
DATA_ROOT_MSMD = '/media/matthias/Data/msmd/'
```

Model Training
--------------

The python script *run_train.py* allows to train all our retrieval models.
You can either train individual models our train all models of one split
at once using the additional shell script *train_models.sh*:

```
./train_models.sh cuda0 models/mutopia_ccal_cont.py <path-to-sheet-manger>/sheet_manager/sheet_manager/splits/all_split.yaml

# $1 ... the device to train on
# $2 ... the model to train
# $3 ... the train split (data) to use for training
```

If you do not want to reproduce all our results reported in the paper
but train only the best performing model you can do this with the following command:
```
python run_train.py --model models/mutopia_ccal_cont.py --data mutopia --train_split <path-to-sheet-manger>/sheet_manager/sheet_manager/splits/all_split.yaml --config exp_configs/mutopia_full_aug.yaml
```
This command trains a model on the all-split (containing pieces of all different composers)
in the full data augmentation setting.<br>
Once this is done there is one final step missing.
As the CCA-Projection-Layer is based on the internal statistics of the training batches
we fine tune it with a very large batch (here 25000 samples) to push the model
to its best performance:
```
python refine_cca.py --n_train 25000 --model models/mutopia_ccal_cont.py --data mutopia --train_split <path-to-sheet-manger>/sheet_manager/sheet_manager/splits/all_split.yaml --config exp_configs/mutopia_full_aug.yaml
```
Once this is done you should end up with fairly well performing
mutimodal audio-sheet music encoders.


Evaluation (Snippet / Excerpt Retrieval)
----------------------------------------
The code structure of the evaluation part of the repository is in line with the training
function.
So in order to evaluate all models of a certain split you can either call:
```
./eval_models.sh cuda0 models/mutopia_ccal_cont_rsz.py <path-to-sheet-manger>/sheet_manager/sheet_manager/splits/all_split.yaml
# $1 ... the device to evaluate on
# $2 ... the model to train
# $3 ... the train split (data) to use for evaluation
```
All results will be printed to your command line output
and in addition dumped to the model folder in your <EXP_ROOT> directory (if the flag *--dump_results* is active).<br>
Again, you can also evaluate a model individually:
```
python run_eval.py --dump_results --model models/mutopia_ccal_cont.py --data mutopia --train_split <path-to-sheet-manger>/sheet_manager/sheet_manager/splits/all_split.yaml --config exp_configs/mutopia_full_aug.yaml --estimate_UV --n_test 2000
```
By adding the flag
```
# (audio-query - to - sheet music)
--V2_to_V1
```
you can change the retrieval direction to audio-query - to - sheet music.
If the flag is not present we retrieve audio (spectrogram excerpts) from image queries.


Evaluation (Score / Performance) Identification
-----------------------------------------------
To reporduce our experiments on score and performance identification you can
run the following sheel script.
```
./eval_piece_retrieval.sh cuda0 models/mutopia_ccal_cont_rsz.py <path-to-sheet-manger>/sheet_manager/sheet_manager/splits/all_split.yaml
# $1 ... the device to evaluate on
# $2 ... the model to train
# $3 ... the train split (data) to use for evaluation
```
Again, you can also run these experiments individually by calling:
```
python audio_sheet_server.py --model models/mutopia_ccal_cont.py --full_eval --init_sheet_db --estimate_UV --dump_results --train_split <path-to-sheet-manger>/sheet_manager/sheet_manager/splits/all_split.yaml --config --config exp_configs/mutopia_full_aug.yaml
```
for score identification from audio recordings and
```
python sheet_audio_server.py --model models/mutopia_ccal_cont.py --full_eval --init_audio_db --estimate_UV --dump_results --train_split <path-to-sheet-manger>/sheet_manager/sheet_manager/splits/all_split.yaml --config --config exp_configs/mutopia_full_aug.yaml
```
for finding performances given a certain score as a query.
The flags
```
--init_sheet_db
--init_audio_db
```
indicate weather to create a database of sheet snippets or audio excerpts or to
load a precomputed database from the disk.
