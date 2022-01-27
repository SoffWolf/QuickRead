#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=ppo_train.out

module purge
module load miniconda
#module load git-lfs
source activate torch
export HF_HOME=$WRKDIR/hf_home 

CUDA_LAUNCH_BLOCKING=1 python3 training.py

