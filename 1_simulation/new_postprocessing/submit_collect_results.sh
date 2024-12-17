#!/bin/bash

#SBATCH --qos=priority
#SBATCH --job-name=dsv2res
#SBATCH --output=logs/%x-%j-%N.out
#SBATCH --error=logs/%x-%j-%N.err
#SBATCH --ntasks=1
#SBATCH --time=1-0


echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

/home/nauck/software/Julia/other_versions/julia-1.9.1/bin/julia prepare_dataframes.jl