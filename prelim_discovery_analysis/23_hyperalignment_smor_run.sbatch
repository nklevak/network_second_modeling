#!/bin/bash
#SBATCH --job-name=hyperalign_smor
#SBATCH --output=logs/hyperalign_smor_%j.out
#SBATCH --error=logs/hyperalign_smor_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=russpold

cd /oak/stanford/groups/russpold/users/nklevak/network_second_modeling/prelim_discovery_analysis
mkdir -p logs

/oak/stanford/groups/russpold/users/nklevak/network_second_modeling/.venv/bin/python 23_hyperalignment_smor_run.py
