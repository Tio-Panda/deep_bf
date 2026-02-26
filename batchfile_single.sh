#!/bin/bash

#SBATCH --job-name deep_bf_test
#SBATCH -t 3:30:00
#SBATCH -p batch
#SBATCH -q batch
#SBATCH --cpus-per-task 4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH -o /mnt/workspace/%u/slurm-out/deep_bf_test-%1-vtest.out
#SBATCH --mail-type=END
#SBATCH --mail-user=mailto:sebastian.gutierremi@usm.cl

module load conda
conda activate pytorch
python train.py
