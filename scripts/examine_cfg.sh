#!/usr/bin/env bash

##################################################
##### Additional Instructions for Riselab Researchers
##################################################

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10
#SBATCH  --gres gpu:4

deactivate
source $PYTHON_ENV/conda/bin/activate
conda activate qdtrack
cd $CODE/qd-track
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python tools/print_config.py /home/eecs/monghim.ng/mmdetection/configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py

<< sample_cmd
bash $CODE/qd-track/scripts/train_tao.sh
cksbatch --nodelist=flaminio $CODE/qd-track/scripts/train_tao.sh
sample_cmd
