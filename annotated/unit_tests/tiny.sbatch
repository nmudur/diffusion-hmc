#!/bin/sh
#SBATCH --nodes=1
#SBATCH --mem=12000
#SBATCH --ntasks=4
#SBATCH -t 0-00:10
#SBATCH -p test

#SBATCH -J Sample_5-7
#SBATCH --account=finkbeiner_lab

module load parallel

PROJDIR="/n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/diffusion-models-for-cosmological-fields/annotated/results/"
MODELRUN="Run_5-7_0-50/"
LOGGINGDIR="sample_log/unit_test/"
MEM=2500
runlogfile=$LOGGINGDIR/"runtasks.log"
srunlogfile=$LOGGINGDIR/"log_${SLURM_JOB_ID}_"

condapy="/n/home02/nmudur/.conda/envs/pytorch_a100/bin/python"
srun="srun --exclusive -n1 -N1 --mem-per-cpu=${MEM} "
parallel="parallel --delay .2 -j $SLURM_NTASKS --joblog $runlogfile --resume-failed"
$parallel "$srun tiny.sh $MODELRUN {1} {%} 1> $srunlogfile{1}.log 2>$srunlogfile{1}.err" ::: {100000..500000..16000}
