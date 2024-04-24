#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-06:00
#SBATCH -p fink_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 8000
#SBATCH -o stout/classifier256_%j.o
#SBATCH -e sterr/classifier256_%j.e

module load Anaconda3/2020.11
module load CUDA/10.0.130

source activate pytorch_a100
/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python noisy_field_classifier.py ../config/classifier.yaml
