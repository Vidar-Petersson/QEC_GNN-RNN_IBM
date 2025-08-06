#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-431      # Project
#SBATCH -p alvis                # Partition
#SBATCH -t 1-00:00:00           # time limit days-hours:minutes:seconds
#SBATCH -o logs/out_%A.out
#SBATCH --gpus-per-node=A40:1 # GPUs 64GB of RAM; cost factor 1.0

# send script
apptainer exec bash_container.sif python examples/train_nn.py