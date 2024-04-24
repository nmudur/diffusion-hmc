#!/bin/bash
#SBATCH --account=finkbeiner_lab
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-04:00
#SBATCH -p fink_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 12000
#SBATCH -o stout/few64_try0_%j.o
#SBATCH -e sterr/few64_try0_%j.e

module load Anaconda3/2020.11
module load CUDA/10.0.130

source activate pytorch_a100

/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python main.py config/e1_nx64_newlinearsmall.yaml
#/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python main.py config/e1_nx64_final.yaml
#/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python main.py config/e1_nx256_final.yaml
#/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python main_ddp.py config/e1_nx64_ddp.yaml
#/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python train_wdecoder.py config/wdecoder256.yaml
