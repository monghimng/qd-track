#!/usr/bin/env bash

##################################################
##### Additional Instructions for Riselab Researchers
##################################################

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10
#SBATCH  --gres gpu:2

deactivate
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

NAME=freeze_lvis_0
NAME=val_on_train
#mkdir -p work_dirs/$NAME

#python tools/train.py \
PORT=294006 bash ./tools/dist_train.sh \
configs/qdtrack-frcnn_r50_fpn_12e_tao.py \
2 \
--work-dir work_dirs/$NAME \
--cfg-options \
data.samples_per_gpu=10 \
data.workers_per_gpu=3 \
evaluation.interval=2 \
load_from=pretrained/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis-dbd06831.pth \
total_epochs=20 \
#--resume-from=pretrained/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis-dbd06831.pth \
#--resume-from=pretrained/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth \
#data.test.ann_file=data/tao/annotations_coco/small_validation.json \
#data.val.ann_file=data/tao/annotations_coco/small_validation.json \
#3 \
#data.workers_per_gpu=3 \
#configs/qdtrack-frcnn_r50_fpn_12e_bdd100k.py \
#$DATA/qd-track/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed_2.pth \
#[optional arguments]



<< sample_cmd
bash $CODE/qd-track/scripts/train_tao.sh
cksbatch --nodelist=flaminio $CODE/qd-track/scripts/train_tao.sh
sample_cmd
