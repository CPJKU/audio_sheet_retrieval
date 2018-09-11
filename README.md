
# Learning Audio-Sheet Music Correspondences for Cross-Modal Retrieval and Piece Identification
This repository contains all experimental code for reproducing the results
reported in our article:

>[Learning Audio-Sheet Music Correspondences for Cross-Modal Retrieval and Piece Identification](https://transactions.ismir.net/articles/10.5334/tismir.12/)
([PDF](https://transactions.ismir.net/articles/10.5334/tismir.12/galley/8/download/)).<br>
Dorfer M., Hajič J. jr., Arzt A., Frostel H., and Widmer G.<br>
*Transactions of the International Society for Music Information Retrieval*, 2018

The paper above is an invited extension of the work presented in:

>[Learning audio-sheet music correspondences for score identification and offline alignment](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/32_Paper.pdf).<br>
Dorfer M., Arzt A., and Widmer G.<br>
In *Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)*, 2017.

The retrieval methodology employed in both works is based on
the *CCA Projection Layer* described in:

>[End-to-End Cross-Modality Retrieval with CCA Projections and Pairwise Ranking Loss.](https://link.springer.com/article/10.1007/s13735-018-0151-5)<br>
Dorfer M., Schlüter J., Vall A., Korzeniowski F., and Widmer G.<br>
International Journal of Multimedia Information Retrieval, 2018

An implementation of the cca layer is contained in this repository and also available [here](https://github.com/CPJKU/cca_layer).

# Table of Contents
  * [Setup and Requirements](#installation)
  * [MSMD Data Set](#msmd)
  * [Preparation](#preparation)
  * [Tutorials](#tutorials)
  * [Model Training](#training)
  * [Model Evaluation](#evaluation)

# Setup and Requirements <a id="installation"></a>
For a list of required python packages see the *requirements.txt*
or just install them all at once using pip.
```
pip install -r requirements.txt
```

We also provide an [anaconda environment file](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
which can be installed as follows:
```
conda env create -f environment.yaml
```

To install the *audio_sheet_retrieval* package in develop mode (this is what we recommend) run
```
python setup.py develop --user
```
in the root folder of the package.

You will also need the **MSMD dataset python package** available
at your system in order to be able to load the data (see below how you get it).

# The MSMD Data Set <a id="msmd"></a>
Almost all of our experiments are based on the proposed Mulitmodal Sheet Music Data Set (MSMD).
For a detailed description of the MSMD data and how to get and load it please visit our
[data set repository](https://github.com/CPJKU/msmd).
The only set of experiments not covered in this repository are the ones carried out
on commercially licenced sheet music.
However, all our models are trained exclusively on MSMD.

# Preparation <a id="preparation"></a>
Before you can start running the code make sure that all paths are configured correctly.
In particular, you have to specify two paths in the file *audio_sheet_retrieval/config/settings.py*:
```
# path where model folder gets created and parameters and results get dumped
EXP_ROOT = "/home/matthias/experiments/audio_sheet_retrieval/"
# path where to find the data (In our case the root directory of the MSMD dataset)
DATA_ROOT_MSMD = '/media/matthias/Data/msmd/'
```
Once this is done we can start training and evaluating our retrieval models.

# Tutorials <a id="tutorials"></a>
If you just want to apply our models to your own sheet music our audios
check out or tutorials.
So far we provide the following [tutorials as ipython notebooks](tutorials):
 - Embedding Tutorial
 - Embedding Tutorial Audio-to-Audio

# Model Training <a id="training"></a>
The python script *run_train.py* allows you to train all individual retrieval models.
Alternatively, if you would like to train all models of one split
at once you can use the additional shell script *train_models.sh*:

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
in the full data augmentation setting (sheet music and audio augmentation).<br>
Once this is done there is a final step missing.
As the CCA-Projection-Layer is based on the internal statistics of the training batches
we fine tune it with a very large batch (here 25000 samples) to push the model
to its best performance:
```
python refine_cca.py --n_train 25000 --model models/mutopia_ccal_cont.py --data mutopia --train_split <path-to-sheet-manger>/sheet_manager/sheet_manager/splits/all_split.yaml --config exp_configs/mutopia_full_aug.yaml
```
After this step you should end up with fairly well performing mutimodal audio-sheet music encoders.


# Evaluation (Snippet / Excerpt Retrieval) <a id="evaluation"></a>
The code structure of the evaluation part of the repository is in line with the training functionality.
To evaluate all models of a certain split at once simply call:
```
./eval_models.sh cuda0 models/mutopia_ccal_cont_rsz.py <path-to-sheet-manger>/sheet_manager/sheet_manager/splits/all_split.yaml
# $1 ... the device to evaluate on
# $2 ... the model to train
# $3 ... the train split (data) to use for evaluation
```
All results will be printed to your command line output
and in addition dumped to the model folder in your <EXP_ROOT> directory (if the flag *--dump_results* is active).<br>
Again, you can also evaluate the models individually:
```
python run_eval.py --dump_results --model models/mutopia_ccal_cont.py --data mutopia --train_split <path-to-sheet-manger>/sheet_manager/sheet_manager/splits/all_split.yaml --config exp_configs/mutopia_full_aug.yaml --estimate_UV --n_test 2000
```
By adding the flag
```
# (audio-query - to - sheet music)
--V2_to_V1
```
you can change the retrieval direction to *audio-query - to - sheet music*.
If the flag is not present we retrieve audio (spectrogram excerpts) from image queries
by default.


# Evaluation (Score / Performance) Identification
To reproduce our experiments on score and performance identification you can
run the following shell script.
```
./eval_piece_retrieval.sh cuda0 models/mutopia_ccal_cont_rsz.py <path-to-sheet-manger>/sheet_manager/sheet_manager/splits/all_split.yaml
# $1 ... the device to evaluate on
# $2 ... the model to train
# $3 ... the train split (data) to use for evaluation
```
As above, you can also run these experiments individually by calling:
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
