#!/bin/bash

#SBATCH --qos=priority
#SBATCH --job-name=dsv2pytorch
#SBATCH --output=logs/%x-%j-%N.out
#SBATCH --error=logs/%x-%j-%N.err
#SBATCH --ntasks=1
#SBATCH --time=1-0


echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"
module load anaconda
source activate py311ptg
python prepare4pytorch.py
