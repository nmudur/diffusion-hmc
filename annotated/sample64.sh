#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:15
#SBATCH -p fink_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 24000
#SBATCH -o sample_log/exp2_nx64_%j.o
#SBATCH -e sample_log/exp2_nx64_%j.e

module load Anaconda3/2020.11
module load CUDA/10.0.130
source activate pytorch_a100
#################
PROJDIR="/n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/diffusion-models-for-cosmological-fields/annotated/results/"
MODELRUN="Run_5-16_17-5/"
SAMPDIR="$PROJDIR/checkpoint64_samples/$MODELRUN"
echo $SAMPDIR


for CKPNUM in {44000..60000..4000}
do
    CKPFILE="checkpoint_$CKPNUM.pt"
    CHECKPOINT="$PROJDIR/samples_exps/$MODELRUN/$CKPFILE"
    echo $CHECKPOINT
    /n/home02/nmudur/.conda/envs/pytorch_a100/bin/python sample_single_checkpoint.py --img_size 64 --checkpoint $CHECKPOINT --savedir $SAMPDIR --num_params 10 --num_samples 5 --batchsize 120
done
