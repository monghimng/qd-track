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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

##################################################
##### Instructions for General Researchers
##################################################

#CUDA_LAUNCH_BLOCKING=1 \
#python tools/train.py \
#configs/qdtrack-frcnn_r50_fpn_12e_tao.py \
#--cfg-options \
#data.samples_per_gpu=1 \
#data.workers_per_gpu=0 \

#PORT=295005 \
#bash ./tools/dist_train.sh \
python tools/train.py \
configs/qdtrack-frcnn_r50_fpn_12e_tao.py \
--cfg-options \
data.samples_per_gpu=9 \
data.workers_per_gpu=3 \
#3 \
#data.workers_per_gpu=3 \
#configs/qdtrack-frcnn_r50_fpn_12e_bdd100k.py \
#$DATA/qd-track/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed_2.pth \
#[optional arguments]



<< sample_cmd
bash $CODE/qd-track/scripts/train_tao.sh
cksbatch --nodelist=flaminio $CODE/qd-track/scripts/train_tao.sh
sample_cmd
