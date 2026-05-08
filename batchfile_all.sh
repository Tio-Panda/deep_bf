#!/bin/bash

#SBATCH --job-name deep_bf
#SBATCH -t 5:00:00
#SBATCH -p batch
#SBATCH -q batch
#SBATCH --cpus-per-task 4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH -o /mnt/workspace/%u/slurm-out/deep_bf-%a.out
#SBATCH --mail-type=END
#SBATCH --mail-user=mailto:sebastian.gutierremi@usm.cl
#SBATCH --array=1-6

module load conda
conda activate pytorch

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" params.txt)

read -r LOCATION EXPERIMENT_ID<<< "$LINE"

python train.py \
    -location "$LOCATION" \
    -e_id "$EXPERIMENT_ID"
