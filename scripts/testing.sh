#!/usr/bin/env bash

##################################################
##### Additional Instructions for Riselab Researchers
##################################################

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10
#SBATCH  --gres gpu:3

source $PYTHON_ENV/conda/bin/activate
conda activate qdtrack
cd $CODE/qd-track
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

##################################################
##### Instructions for General Researchers
##################################################
# debug using bdd
python tools/test.py \
configs/qdtrack-frcnn_r50_fpn_12e_bdd100k.py \
pretrained/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth \
--eval track \
--cfg-options \
data.test.ann_file=data/bdd/tracking/annotations/small_bdd100k_track_val_taoformat.json \
data.val.ann_file=data/bdd/tracking/annotations/small_bdd100k_track_val_taoformat.json \

exit

#bash ./tools/dist_test.sh \
python tools/test.py \
configs/qdtrack-frcnn_r50_fpn_12e_tao.py \
/data/ck/qd-track/work_dirs/qdtrack-frcnn_r50_fpn_12e_tao/epoch_2.pth \
--eval track \
--cfg-options \
data.test.ann_file=data/tao/annotations_coco/small_validation.json \
data.val.ann_file=data/tao/annotations_coco/small_validation.json \
#data.test.key_img_sampler.interval=5000 \
#configs/qdtrack-frcnn_r50_fpn_12e_bdd100k.py \
#pretrained/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed_2.pth \
#3 \
#--eval bbox \
#--show \
#${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show] [--cfg-options]



<< sample_cmd
bash $CODE/qd-track/scripts/testing.sh
cksbatch --nodelist=flaminio $CODE/qd-track/scripts/testing.sh
sample_cmd
