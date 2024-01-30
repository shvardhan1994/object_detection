#!/bin/bash

#SBATCH -o slurm_out_%j.job
#SBATCH -e slurm_error_%j.job
#SBATCH -p develbooster 
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --nodes 1
#SBATCH --mem=30G
#SBATCH -t 02:00:00
#SBATCH --nice=10000
#SBATCH --account=hai_hmgu



echo 'Starting script'

source ../sc_venv_template/activate.sh
HYDRA_FULL_ERROR=1 python train.py
