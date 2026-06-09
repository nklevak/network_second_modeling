#!/bin/bash
#SBATCH --job-name=hyperalign_surf_run
#SBATCH --output=../logs/hyperalign_surf_run_%j.out
#SBATCH --error=../logs/hyperalign_surf_run_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=russpold

cd /oak/stanford/groups/russpold/users/nklevak/network_second_modeling/discovery_surface_analysis
mkdir -p ../logs

/oak/stanford/groups/russpold/users/nklevak/network_second_modeling/.venv/bin/python 23_hyperalignment_surface_run.py
