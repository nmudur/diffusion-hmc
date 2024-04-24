#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -p siag_gpu
#SBATCH --account=siag_lab 
#SBATCH --gres=gpu:1
#SBATCH --mem 24000
#SBATCH -o stout/few256_final_%j.o
#SBATCH -e sterr/few256_final_%j.e

module load python
module load cuda

source activate rocky4

/n/home02/nmudur/.conda/envs/rocky4/bin/python main.py config/e1_nx256_newlinearsmall.yaml
