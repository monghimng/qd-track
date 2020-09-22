#!/usr/bin/env bash

##################################################
##### Additional Instructions for Riselab Researchers
##################################################

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10

REPO=mmdetection
mkdir -p $CODE/$REPO $DATA/$REPO

############################## installing conda
cd $PYTHON_ENV
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh
# when prompted, install it at $PYTHON_ENV/conda

source $PYTHON_ENV/conda/bin/activate
conda create -n qdtrack python=3.7 -y
conda activate qdtrack
conda install pytorch=1.6 torchvision cudatoolkit=10.2 -c pytorch
pip install mmcv-full==latest+torch1.6.0+cu102 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
cd $CODE/mmdetection/
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
pip install pandas tqdm wandb

cd $CODE/qd-track
python setup.py develop

# getting data to their format
python tools/convert_datasets/bdddet2coco.py -i /data/ck/data/bdd/detection/labels -o /data/ck/data/bdd/detection/annotations
python tools/convert_datasets/bddtrack2cocovid.py -i /data/ck/data/bdd/tracking/labels-20 -o /data/ck/data/bdd/tracking/annotations
cd detection/images/
mv 100k/* ./

ln -s $DATA/data $CODE/qd-track/data

<< sample_cmd
bash ~/mmdetection/scripts/setup_on_rise_machiens.sh
deactivate;
source $PYTHON_ENV/conda/bin/activate
conda activate qdtrack
sample_cmd
