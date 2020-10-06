#!/usr/bin/env bash

##################################################
##### Additional Instructions for Riselab Researchers
##################################################

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10
#SBATCH  --gres gpu:2

source $PYTHON_ENV/conda/bin/activate
conda activate qdtrack
cd $CODE/qd-track
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

##################################################
##### Instructions for General Researchers
##################################################

#python tools/test.py \
PORT=235008 bash ./tools/dist_test.sh \
configs/qdtrack-frcnn_r50_fpn_12e_tao.py \
pretrained/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis-dbd06831.pth \
2 \
--eval bbox \
--cfg-options \
test_cfg.rcnn.score_thr=0.0001 \
test_cfg.rcnn.max_per_img=300 \
#data.test.ann_file=data/tao/annotations_coco/small_validation.json \
#data.test.ann_file=data/tao/annotations_coco/validation.json \
#--gpu-collect \
#work_dirs/qdtrack-frcnn_r50_fpn_12e_tao/epoch_2.pth \

exit

# test_cfg configs for lvis
test_cfg.rcnn.score_thr=0.0001 \
test_cfg.rcnn.max_per_img=300 \

<< sample_cmd
bash $CODE/qd-track/scripts/test_tao.sh
cksbatch --nodelist=flaminio $CODE/qd-track/scripts/test_tao.sh
sample_cmd
