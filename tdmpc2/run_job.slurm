#!/bin/bash
#SBATCH --job-name=tdmpc2_array
#SBATCH --output=logs/tdmpc2_%A_%a.out
#SBATCH --error=logs/tdmpc2_%A_%a.err
#SBATCH --array=0-24     # 25 jobs for 5x5 alpha/beta combos
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --account=def-rhinehar

module load apptainer

# Sweep values for alpha and beta
values=(-0.01 -0.1 0 0.01 0.1)

# Map SLURM array index to (alpha_index, beta_index)
alpha_index=$((SLURM_ARRAY_TASK_ID / 5))
beta_index=$((SLURM_ARRAY_TASK_ID % 5))

alpha=${values[$alpha_index]}
beta=${values[$beta_index]}

export WANDB_API_KEY=8a8ce7fbc7a816639a48369e34b70f2503528d06
export WANDB_MODE=offline
export WANDB_DIR=/mnt/tdmpc2/wandb_logs

SIF_PATH=/home/nabail/projects/def-rhinehar/nabail/ubp.sif
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
    python3 '$SCRIPT_PATH' +dyn_uncer_beta_coef_value=$beta +rew_uncer_alpha_coef_value=$alpha
  "
