#!/bin/bash
#SBATCH --job-name=skinlesion_cnn
#SBATCH --output=/blue/bme6938/Josephtsenum/Project2/logs/slurm_%j.out
#SBATCH --error=/blue/bme6938/Josephtsenum/Project2/logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=02:00:00
#SBATCH --account=bme6938

# ══════════════════════════════════════════════════════════════════════════════
# Project 2 — CNN Skin Lesion Classification
# BME6938 Medical AI · Group 6
#
# Usage:
#   cd /blue/bme6938/Josephtsenum/Project2
#   sbatch run_training.sh
# ══════════════════════════════════════════════════════════════════════════════

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Load modules
module load conda
conda activate /blue/bme6938/share/envs/medai  # Adjust to your env name

# Create output dirs
mkdir -p /blue/bme6938/Josephtsenum/Project2/{models,figures,logs}

# Run notebook as script
cd /blue/bme6938/Josephtsenum/Project2
jupyter nbconvert --to script Project2_SkinLesion_CNN.ipynb --stdout | python3

echo "Job completed: $(date)"
