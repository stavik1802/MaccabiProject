#!/bin/bash
#SBATCH --job-name=mpac_train
#SBATCH --output=logs/mpac_train_%j.out
#SBATCH --error=logs/mpac_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00  # Run up to 12 hours overnight
#SBATCH --partition=all  # Replace with your cluster's partition name
#SBATCH --gres=gpu:1        # If you need a GPU, remove if CPU-only
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stavos114@gmail.com

# Optional: Load required modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Maccabi

# Ensure logs directory exists
mkdir -p logs


# Run your training script
python -u pre_proccess.py run1

