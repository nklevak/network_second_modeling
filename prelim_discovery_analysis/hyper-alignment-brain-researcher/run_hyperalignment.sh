#!/bin/bash
#SBATCH --job-name=hyperalignment
#SBATCH --output=hyperalignment_%j.out
#SBATCH --error=hyperalignment_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=russpold

source /home/users/nklevak/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /oak/stanford/groups/russpold/users/nklevak/network_second_modeling/prelim_discovery_analysis
python 21_hyperalignment_run.py
