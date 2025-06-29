#!/bin/bash
#SBATCH --job-name=10
#SBATCH --output=logs/tdmpc2_%j.out
#SBATCH --error=logs/tdmpc2_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --account=def-rhinehar

module load apptainer

# W&B config
export WANDB_API_KEY=8a8ce7fbc7a816639a48369e34b70f2503528d06
export WANDB_MODE=offline
export WANDB_DIR=/mnt/tdmpc2/wandb_logs
mkdir -p /home/nabail/projects/def-rhinehar/nabail/tdmpc2/wandb_logs

# Paths
SIF_PATH=/home/nabail/projects/def-rhinehar/nabail/ubp.sif
SCRIPT_PATH=/mnt/tdmpc2/tdmpc2/train.py  # container-side path

# Run script inside Apptainer
apptainer exec --nv \
  --no-home \
  --fakeroot \
  --bind /home/nabail/projects/def-rhinehar/nabail/tdmpc2:/mnt/tdmpc2 \
  "$SIF_PATH" \
  bash --login -c "
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    conda activate ubp
    cd /mnt/tdmpc2
    export MUJOCO_GL=disable
    export WANDB_MODE=offline
    export WANDB_DIR=/mnt/wandb_logs
    python3 \"$SCRIPT_PATH\"
  "
