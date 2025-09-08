#!/bin/bash
#SBATCH --job-name=tdmpc2_array
#SBATCH --output=logs/tdmpc2_%A_%a.out
#SBATCH --error=logs/tdmpc2_%A_%a.err
#SBATCH --array=1-5       # 5 jobs, seeds 1 to 5
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --account=def-rhinehar

module load apptainer

# SLURM_ARRAY_TASK_ID will be used as the seed
seed=$SLURM_ARRAY_TASK_ID

export WANDB_API_KEY=8a8ce7fbc7a816639a48369e34b70f2503528d06
export WANDB_MODE=offline
export WANDB_DIR=/mnt/tdmpc2/wandb_logs

SIF_PATH=/home/nabail/projects/def-rhinehar/nabail/ubp2.sif
SCRIPT_PATH=/mnt/tdmpc2/tdmpc2/train.py

apptainer exec --nv \
  --no-home \
  --fakeroot \
  --bind /home/nabail/projects/def-rhinehar/nabail/tdmpc2:/mnt/tdmpc2 \
  "$SIF_PATH" \
  bash --login -c "
    source \$(conda info --base)/etc/profile.d/conda.sh
    conda activate ubp
    cd /mnt/tdmpc2
    export MUJOCO_GL=disable
    export WANDB_MODE=offline
    export WANDB_DIR=/mnt/wandb_logs
    python3 '$SCRIPT_PATH' +seed_value=$seed
  "