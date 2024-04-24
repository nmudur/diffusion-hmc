#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 1-00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 8000
#SBATCH -o stout/output128_try0_%j.o
#SBATCH -e sterr/error128_try0_%j.e

module load Anaconda3/2020.11
module load CUDA/10.0.130

source activate pytorch_env2
/n/home02/nmudur/.conda/envs/pytorch_env2/bin/python main.py config/params128_bl.yaml

