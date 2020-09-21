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

##################################################
##### Instructions for General Researchers
##################################################

#python tools/test.py \
bash ./tools/dist_test.sh \
configs/qdtrack-frcnn_r50_fpn_12e_bdd100k.py \
$DATA/qd-track/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed_2.pth \
3 \
--eval track \
#--eval bbox \
#--show \
#${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show] [--cfg-options]



<< sample_cmd
bash $CODE/qd-track/scripts/testing.sh
cksbatch --nodelist=flaminio $CODE/qd-track/scripts/testing.sh
sample_cmd
