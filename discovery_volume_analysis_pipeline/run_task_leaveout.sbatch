#!/bin/bash
#SBATCH --job-name=task_leaveout_align
#SBATCH --partition=russpold
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/task_leaveout_%j.out
#SBATCH --error=logs/task_leaveout_%j.err

mkdir -p logs

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

source /oak/stanford/groups/russpold/users/nklevak/network_second_modeling/.venv/bin/activate

python run_task_leaveout.py

echo "Job finished: $(date)"
