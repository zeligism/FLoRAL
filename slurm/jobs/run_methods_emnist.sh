#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --partition=long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:4
#SBATCH --job-name=run_methods_emnist
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --time=4320

FORCE_RUN_LOCALLY=1 python scripts/run_experiment.py run_methods emnist
