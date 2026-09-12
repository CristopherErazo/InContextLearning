#!/bin/bash
#SBATCH --job-name=icl
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# --account comes from $SBATCH_ACCOUNT. 8 CPUs per GPU is the billing ratio:
# 8/32 == 1/4 == the GPU fraction, so the GPU sets the rate and the CPUs
# are effectively free.  Cost: 8 CPUh per wallclock hour.
#
# Usage:  sbatch train.sh                        # defaults
#         sbatch train.sh lr=3e-4 seed=1         # omegaconf overrides pass through

set -euo pipefail

source "$HOME/.leonardo_env.sh"

# Wheels ship their own CUDA runtime; loading modules risks binding torch to a
# mismatched system cuDNN/NCCL.  Compute nodes have no internet, so make any
# stray network call fail fast instead of burning wallclock on timeouts.
module purge
export UV_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PROJ="$HOME/projects/in-context-learning/code"
cd "$PROJ"
source .venv/bin/activate

RUN_NAME="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
BASE_DIR="$RUNS/$SLURM_JOB_NAME"
mkdir -p "$BASE_DIR" logs

echo "node     : $(hostname)   gpu: ${CUDA_VISIBLE_DEVICES:-none}"
echo "run      : $RUN_NAME"
echo "base_dir : $BASE_DIR"
python -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())'

srun python scripts/train.py \
     extra_args.experiment_name="$RUN_NAME" \
     "$@"
     # extra_args.base_dir="$BASE_DIR" \
     # "$@"

echo "finished : $(date '+%F %T')"
