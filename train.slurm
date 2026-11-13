#!/bin/bash
#SBATCH --job-name=processdatabase
#SBATCH --array=1
#SBATCH --time=00:30:00               # Max runtime (HH:MM:SS) - adjust if Minion takes longer
#SBATCH --mem=8G                       # RAM per job
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%a.out      
#SBATCH --error=logs/%a.err       

# Create a logs directory if it doesn't exist so your folder doesn't get messy
mkdir -p logs

source .venv/bin/activate

python ./train.py  
