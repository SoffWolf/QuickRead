#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --output=upload_n_test_reddit_clean.out

module purge
module load miniconda
module load git-lfs
source activate torch
export HF_HOME=$WRKDIR/hf_home
#pip3 install -r requirements.txt

python3 upload_n_test_reddit_clean.py \
