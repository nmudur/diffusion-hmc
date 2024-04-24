#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:10
#SBATCH -p shared
#SBATCH --mem 4000
#SBATCH -o sample_log/condlik_%j.o
#SBATCH -e sample_log/condlik_%j.e


module load Anaconda2/2019.10-fasrc01
# module load Anaconda3/2020.11
module load CUDA/10.0.130
source activate pytorch_a100

dt="0923_1719"
split="validation"
MODELRUN="Run_5-7_0-50"
CKPNUM=260000
SDPATH="results/samples_exps/$MODELRUN/checkpoint_$CKPNUM.pt"
SAVEDIR="CondLik/$MODELRUN/$split/checkpoint_$CKPNUM/$dt/"
PLOTSDIR="$SAVEDIR/plots/"

/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python compute_likelihoods.py --sdpath $SDPATH --savedir $SAVEDIR --plotsdir $PLOTSDIR --data_subtype $split --pseed 42
