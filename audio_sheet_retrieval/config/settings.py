
import os

# set paths
EXP_ROOT = "/home/stefanb/dev/audio_sheet_retrieval/experiments"
DATA_ROOT_MSMD = '/home/stefanb/dev/msmd_aug/'

# get hostname
hostname = os.uname()[1]

# adopted paths
if hostname in ["rechenknecht0.cp.jku.at", "rechenknecht1.cp.jku.at"]:
    EXP_ROOT = "/home/matthias/experiments/audio_sheet_retrieval/"
    DATA_ROOT_MSMD = '/home/matthias/shared/datasets/msmd_aug/'

elif hostname == "mdhp":
    EXP_ROOT = "/home/matthias/experiments/audio_sheet_retrieval/"
    DATA_ROOT_MSMD = '/media/matthias/Data/Data/msmd/'
