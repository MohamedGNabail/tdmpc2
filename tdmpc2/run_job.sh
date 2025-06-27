#!/bin/bash
#SBATCH --job-name=tdmpc2_seed_1_0_1
#SBATCH --output=logs/tdmpc2_%j.out
#SBATCH --error=logs/tdmpc2_%j.err
#SBATCH --time=0:08:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --account=def-rhinehar

# Load Apptainer
module load apptainer

# Paths
PROJECT_DIR=/home/nabail/projects/def-rhinehar/nabail/tdmpc2/tdmpc2
SIF_PATH=/home/nabail/projects/def-rhinehar/nabail/ubp_app.sif
SCRIPT_PATH=/home/nabail/projects/def-rhinehar/nabail/tdmpc2/tdmpc2/train.py

# Run script inside Apptainer with environment activation
apptainer exec --nv \
  --bind /home/nabail/projects/def-rhinehar/nabail:/mnt \
  "$SIF_PATH" \
  bash -c "
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    conda activate ubp
    export MUJOCO_GL=egl
    python3 "$SCRIPT_PATH"
  "
