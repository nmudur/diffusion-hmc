#!/bin/bash
#SBATCH -J hmc-single
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p siag_gpu
#SBATCH --account=siag_lab
#SBATCH --output=stout/hmc/hmc21_%A_%a.o
#SBATCH --error=sterr/hmc/hmc21_%A_%a.e
#SBATCH --array=0,1,2,5,8

#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=00:50:00

module load python
module load gcc
module load cuda

echo "Running on node: $(hostname)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

/n/home02/nmudur/.conda/envs/rocky4_hmc/bin/python hmc_inference.py 3 $SLURM_ARRAY_TASK_ID

