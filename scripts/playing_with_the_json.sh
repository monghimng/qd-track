#!/usr/bin/env bash

##################################################
##### Additional Instructions for Riselab Researchers
##################################################

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10
#SBATCH  --gres gpu:4

source $PYTHON_ENV/conda/bin/activate
conda activate qdtrack
cd $CODE/qd-track

##################################################
##### Instructions for General Researchers
##################################################

import json
fp = open('/data/datasets/tao/annotations/train.json')
j = json.load(fp)


<< sample_cmd
bash $CODE/qd-track/scripts/train.sh
cksbatch --nodelist=flaminio $CODE/qd-track/scripts/train.sh
sample_cmd
