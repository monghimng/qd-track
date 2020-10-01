#!/usr/bin/env bash

##################################################
##### Additional Instructions for Riselab Researchers
##################################################

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10
#SBATCH  --gres gpu:1

source $PYTHON_ENV/conda/bin/activate
conda activate qdtrack
cd $CODE/qd-track

##################################################
##### Instructions for General Researchers
##################################################

#PORT=295002 \
#bash ./tools/dist_train.sh \
python tools/train.py \
configs/qdtrack-frcnn_r50_fpn_12e_bdd100k.py \
--cfg-options \
data.samples_per_gpu=10 \
data.workers_per_gpu=3 \
data.test.ann_file=data/bdd/tracking/annotations/small_bdd100k_track_val_cocoformat.json \
data.val.ann_file=data/bdd/tracking/annotations/small_bdd100k_track_val_cocoformat.json \
#4 \
#$DATA/qd-track/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed_2.pth \
#[optional arguments]



<< sample_cmd
bash $CODE/qd-track/scripts/train.sh
cksbatch --nodelist=flaminio $CODE/qd-track/scripts/train.sh
sample_cmd
