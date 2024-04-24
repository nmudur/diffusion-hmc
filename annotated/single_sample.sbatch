#!/bin/sh
#SBATCH -t 0-01:00
#SBATCH -p iaifi_gpu
#SBATCH -o sample_log/logs/singlesamp_try0_%j.o
#SBATCH -e sample_log/logs/singlesamp_try0_%j.e

#SBATCH -J Sample_5-7
#SBATCH --account=finkbeiner_lab

#SBATCH --mail-user=nmudur@cfa.harvard.edu
#SBATCH --mail-type=END,FAIL


PROJDIR="/n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/diffusion-models-for-cosmological-fields/annotated/results/"
MODELRUN="Run_5-7_0-50/"
dt=$(date +'%m%d_%H%M')
LOGGINGDIR="sample_log/$MODELRUN/$dt"
if [ ! -d "$LOGGINGDIR" ]; then
    mkdir -p "$LOGGINGDIR"
    echo "Directory '$LOGGINGDIR' created."
else
    echo "Directory '$LOGGINGDIR' already exists."
fi
MEM=20000
srunlogfile=$LOGGINGDIR/"log_${SLURM_JOB_ID}_"

srun="srun -n1 -N1 --mem-per-cpu=${MEM} --gres=gpu:1"
CKPNUM=260000
$srun sample_singlerun_cmd.sh $MODELRUN $CKPNUM 1 $dt 1> $srunlogfile{$CKPNUM}.log 2>$srunlogfile{$CKPNUM}.err
