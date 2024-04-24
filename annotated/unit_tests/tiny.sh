#!/bin/bash
PROJDIR="/n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/diffusion-models-for-cosmological-fields/annotated/results/"
MODELRUN=$1
CKPFILE="checkpoint_$2.pt"
CHECKPOINT="$PROJDIR/samples_exps/$MODELRUN/$CKPFILE"

SAMPDIR="$PROJDIR/checkpoint_samples/$MODELRUN"
echo $CHECKPOINT
echo $SAMPDIR
echo $3
DEVICE_ID=$(($3-1))
echo $DEVICE_ID

module load Anaconda3/2020.11
module load CUDA/10.0.130
source activate pytorch_a100

/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python tinytest.py --checkpoint $CHECKPOINT --savedir $SAMPDIR --dataset train --num_params 10 --num_samples 5 --device_id $DEVICE_ID
/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python tinytest.py --checkpoint $CHECKPOINT --savedir $SAMPDIR --dataset validation --num_params 10 --num_samples 5 --device_id $DEVICE_ID
