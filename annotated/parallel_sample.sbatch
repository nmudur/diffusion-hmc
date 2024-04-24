#!/bin/sh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -t 0-02:00
#SBATCH -p siag_gpu
#SBATCH --mem=20G
#SBATCH -o /n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/diffusion-models-for-cosmological-fields/annotated/results/samples_exps/Logs/log_%j.o
#SBATCH -e /n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/diffusion-models-for-cosmological-fields/annotated/results/samples_exps/Logs/log_%j.e
#SBATCH -J Sample_LHfields
#SBATCH --account=siag_lab

#SBATCH --mail-user=nmudur@cfa.harvard.edu
#SBATCH --mail-type=END,FAIL

module load ncf/1.0.0-fasrc01
module load parallel/20230422-rocky8_x64-ncf
# Also additionally run the above command in the environment where this script is run

PROJDIR="/n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/diffusion-models-for-cosmological-fields/annotated/results/"
MODELRUN="Run_10-30_2-32/"
dt=$(date +'%m%d_%H%M')
LOGGINGDIR="sample_log/$MODELRUN/$dt"
if [ ! -d "$LOGGINGDIR" ]; then
    mkdir -p "$LOGGINGDIR"
    echo "Directory '$LOGGINGDIR' created."
else
    echo "Directory '$LOGGINGDIR' already exists."
fi
MEM=20000
runlogfile=$LOGGINGDIR/"runtasks_8kinterleave.log"
srunlogfile=$LOGGINGDIR/"log_${SLURM_JOB_ID}_"

condapy="/n/home02/nmudur/.conda/envs/rocky4/bin/python"
srun="srun --exclusive -n1 -N1 --mem-per-cpu=${MEM} --gres=gpu:1"
parallel="parallel --delay .2 -j $SLURM_NTASKS --joblog $runlogfile --resume-failed"
$parallel "$srun sample_singlerun_cmd.sh $MODELRUN {1} {%} $dt 1> $srunlogfile{1}.log 2>$srunlogfile{1}.err" ::: 260000
