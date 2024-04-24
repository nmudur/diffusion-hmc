#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-06:00
#SBATCH -p shared
#SBATCH --mem 8000
#SBATCH -o stout/wstclassifier256_%j.o
#SBATCH -e sterr/wstclassifier256_%j.e

module load Anaconda3/2020.11
module load gcc/7.1.0-fasrc01
module load CUDA/10.0.130

source activate pytorch_func
/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python cosmo_classifier.py ../config/wst_classifier.yaml
