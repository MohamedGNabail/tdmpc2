#!/bin/bash

# Load Apptainer
module load apptainer
export WANDB_API_KEY=8a8ce7fbc7a816639a48369e34b70f2503528d06
# Paths
SIF_PATH=/home/nabail/projects/def-rhinehar/nabail/ubp.sif
HOST_PROJECT_DIR=/home/nabail/projects/def-rhinehar/nabail/tdmpc2/tdmpc2
SCRIPT_PATH=/mnt/tdmpc2/tdmpc2/train.py  # container-side path

# Run script inside Apptainer with environment activation
apptainer exec --nv \
  --bind /home/nabail/projects/def-rhinehar/nabail:/mnt \
  "$SIF_PATH" \
  bash --login -c "
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    conda activate ubp
    export MUJOCO_GL=disable
    python3 \"$SCRIPT_PATH\"
  "




