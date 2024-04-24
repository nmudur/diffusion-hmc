#!/bin/bash
PROJDIR="/n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/diffusion-models-for-cosmological-fields/annotated/results/"
MODELRUN=$1
CKPFILE="checkpoint_$2.pt"
CHECKPOINT="$PROJDIR/samples_exps/$MODELRUN/$CKPFILE"


DEVICE_ID=$(($3-1))  #$CUDA_VISIBLE_DEVICES
SAMPDIR="$PROJDIR/checkpoint_samples/$MODELRUN/$4"
if [ ! -d "$SAMPDIR" ]; then
    mkdir -p "$SAMPDIR"
    echo "Directory '$SAMPDIR' created."
else
    echo "Directory '$SAMPDIR' already exists."
fi
echo $CHECKPOINT
echo $SAMPDIR
echo $DEVICE_ID

#export CUDA_VISIBLE_DEVICES=$DEVICE_ID

module load python
module load cuda
source activate rocky4

/n/home02/nmudur/.conda/envs/rocky4/bin/python sample_single_checkpoint.py --checkpoint $CHECKPOINT --savedir $SAMPDIR --dataset validation --num_params 5 --num_samples 50 --batchsize 100 --param_seed 9 --sample_seed 1997
