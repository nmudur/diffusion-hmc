#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-03:00
#SBATCH -p fink_gpu
#SBATCH --account=siag_lab
#SBATCH --gres=gpu:1
#SBATCH --mem 12000
#SBATCH -o sample_log/condlik_%j.o
#SBATCH -e sample_log/condlik_%j.e

module load python
module load cuda
source activate rocky4

dt=$(date +'%m%d_%H%M')
split="validation"
MODELRUN="Run_10-30_2-32"
VLBT=$(echo $(seq 0 10) | tr ' ' ',') # Change
# VLB_SEEDS=$(echo $(seq 10 18) | tr ' ' ',')
CKPNUM=260000
NDISC=100
SDPATH="results/samples_exps/$MODELRUN/checkpoint_$CKPNUM.pt"
SAVEDIR="CondLik/$MODELRUN/$split/checkpoint_$CKPNUM/$dt/"
PLOTSDIR="$SAVEDIR/plots/"
if [ ! -d "$SAVEDIR" ]; then
    mkdir -p "$SAVEDIR"
    mkdir -p "$PLOTSDIR"
    echo "Directory '$SAVEDIR' created."
else
    echo "Directory '$SAVEDIR' already exists."
fi

/n/home02/nmudur/.conda/envs/rocky4/bin/python compute_likelihoods.py --sdpath $SDPATH --savedir $SAVEDIR --data_subtype $split --ndiscretization $NDISC --pseed 53 --nsamples 1 --transform_seeds 2 --vlb_seeds 9 --num_params 10 --grid_extent 0.1 --vlb_timesteps $VLBT --reseed_over_time --Nbatches 200 --plotsdir $PLOTSDIR
