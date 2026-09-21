#!/bin/bash
#SBATCH --job-name=icl-sweep
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
#
# Scaling sweep on Leonardo. Same grid, seeds and overrides as
# shell/workstation_sweep.sh; the environment preamble below is lifted from
# shell/train.sh and must be kept in sync with it.
#
# Resource model. --account comes from $SBATCH_ACCOUNT. 8 CPUs per GPU is the
# billing ratio (8/32 == 1/4 == the GPU fraction), so the GPU sets the rate and
# the CPUs are free: 8 CPUh per wallclock hour per array task. The model is tiny
# (two linear-attention layers, no MLP), so at small d_model one A100 is nowhere
# near saturated by a single run -- each array task therefore takes one
# configuration and runs its SEEDS concurrently on the same GPU.
#
# CAVEAT, measured 2026-09-21 on the A100-PCIE workstation at the current baseline
# (V=128 L=128 d=512 B=218): one run already saturates the card. Three concurrent
# seeds ran at 41.2 ms/step each = 13.7 ms per step of work, versus 12.5 ms/step
# for one run at a time. Packing gained nothing. When the GPU is saturated the
# billing argument collapses -- 3 seeds on 1 GPU for 3T costs the same as 3 seeds
# on 3 GPUs for T -- and only the wallclock differs, in favour of not packing.
# Packing still pays at d=64/128 (indices 9, 10). If the d=512 tasks prove too slow,
# split seeds into their own array tasks rather than raising --time.
#
# Because the GPU is the contended resource here (SEEDS processes share it) while
# the 8 CPUs sit idle and are billed anyway, batches are generated on the CPU:
# GEN_DEVICE=cpu. That is the opposite of the icl.config default ("auto" -> cuda),
# which was measured on a single run with the GPU to itself. See CLAUDE.md, Data.
#
# Walltime. The grid spans a ~250x range in step budget (index 5, V=32, is 1360
# steps; index 8, V=512, is 342053), so a uniform --time is set by the worst case
# while most tasks finish in minutes. Runs that learn exit at ~T*, i.e. 1/ALPHA_STEPS
# of the budget; --time is insurance against the ones that never reach STOP_ACC.
# 2h does NOT cover index 8 (nor 7, 4) in that failure case. Submit in two groups:
#   sbatch --array=0-3,5,6,9-11 --time=00:45:00 shell/leonardo_sweep.sh
#   sbatch --array=4,7,8        --time=06:00:00 shell/leonardo_sweep.sh
#
# Usage:
#   sbatch shell/leonardo_sweep.sh                    # the whole grid
#   bash   shell/leonardo_sweep.sh --list             # print the grid, submit nothing
#   sbatch --array=0-3 shell/leonardo_sweep.sh        # a subset of configurations
#
# IMPORTANT: --array above must cover 0..N-1 for the N configurations printed by
# `--list`. It is checked at runtime and the task exits loudly if it does not.

set -euo pipefail

# ---------------------------------------------------------------- knobs
EXP="${EXP:-scaling_sweep}"
SEEDS=(${SEEDS:-1 2 3})
ALPHA_STEPS="${ALPHA_STEPS:-20}"     # budget = ALPHA_STEPS x predicted T* (icl.config.predicted_learning_time)
STOP_ACC="${STOP_ACC:-0.75}"         # early exit; the step reached is T*
N_PRINTS="${N_PRINTS:-300}"          # T* resolution = total_steps / N_PRINTS
ALPHA_LR="${ALPHA_LR:-4000}"         # eta_0 for every run; the grid is in the sweep axes
ALPHA_BATCH="${ALPHA_BATCH:-1500}"   # batch-size scale; B = ALPHA_BATCH / (L * frac_solvable)
MASK_1="${MASK_1:-prev}"             # layer-1 mask: "prev" (j == i-1) or "causal" (j < i)
GEN_DEVICE="${GEN_DEVICE:-cuda}"     # batch sampling device; see the resource model above
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"  # "highest" = true fp32, "high" = TF32

# ---------------------------------------------------------------- the grid
# Keep identical to shell/workstation_sweep.sh.
BASE_V=128; BASE_L=128; BASE_D=512
SWEEP_L=(64 256 512 1024)
SWEEP_V=(32 64 256 512)
SWEEP_D=(64 128 512 1024)

# A sub-sweep may contain the baseline value on its own axis (SWEEP_D holds 512 and
# BASE_D is 512), which would run that point twice under the same seeds and give it
# double weight in the scaling fit. Append only new points.
CONFIGS=()
add_config() {
    local c="$1" existing
    for existing in ${CONFIGS[@]+"${CONFIGS[@]}"}; do
        [[ "$existing" == "$c" ]] && return 0
    done
    CONFIGS+=("$c")
}
add_config "$BASE_V $BASE_L $BASE_D"
for L in "${SWEEP_L[@]}"; do add_config "$BASE_V $L $BASE_D"; done
for V in "${SWEEP_V[@]}"; do add_config "$V $BASE_L $BASE_D"; done
for D in "${SWEEP_D[@]}"; do add_config "$BASE_V $BASE_L $D"; done
NCONF=${#CONFIGS[@]}

if [[ "${1:-}" == "--list" ]]; then
    echo "$NCONF configurations (use --array=0-$((NCONF - 1))), ${#SEEDS[@]} seeds each:"
    for i in "${!CONFIGS[@]}"; do
        read -r V L D <<< "${CONFIGS[$i]}"
        printf '  %2d  V=%-5s L=%-5s d=%-5s\n' "$i" "$V" "$L" "$D"
    done
    exit 0
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?this script must be submitted with sbatch --array}"
if (( TASK_ID >= NCONF )); then
    echo "array index $TASK_ID is out of range: only $NCONF configurations exist." >&2
    echo "resubmit with --array=0-$((NCONF - 1))" >&2
    exit 1
fi
read -r V L D <<< "${CONFIGS[$TASK_ID]}"

# ---------------------------------------------------------------- environment
# Wheels ship their own CUDA runtime; loading modules risks binding torch to a
# mismatched system cuDNN/NCCL. Compute nodes have no internet, so make any stray
# network call fail fast instead of burning wallclock on timeouts.
source "$HOME/.leonardo_env.sh"
module purge
export UV_OFFLINE=1
export HF_HUB_OFFLINE=1

# The seeds run concurrently on one GPU, so split the CPUs between them rather
# than letting each process grab all 8 and thrash.
NPAR=${#SEEDS[@]}
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / NPAR ))
(( OMP_NUM_THREADS < 1 )) && export OMP_NUM_THREADS=1

PROJ="$HOME/projects/in-context-learning/code"
cd "$PROJ"
source .venv/bin/activate

LOG_DIR="logs/$EXP"
mkdir -p "$LOG_DIR" logs

echo "node       : $(hostname)   gpu: ${CUDA_VISIBLE_DEVICES:-none}"
echo "array task : $TASK_ID / $((NCONF - 1))"
echo "config     : V=$V  L=$L  d=$D"
echo "seeds      : ${SEEDS[*]}  (concurrent, OMP_NUM_THREADS=$OMP_NUM_THREADS each)"
echo "model      : mask1=$MASK_1  alpha_lr=$ALPHA_LR  alpha_batch=$ALPHA_BATCH"
echo "numerics   : gen_device=$GEN_DEVICE  matmul_precision=$MATMUL_PRECISION"
echo "budget     : alpha_steps=$ALPHA_STEPS  stop_at_accuracy=$STOP_ACC  n_prints=$N_PRINTS"
python -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())'

# ---------------------------------------------------------------- run
# No srun: these are single-process jobs sharing one GPU, and srun would serialise
# them into the single task this allocation has.
pids=()
for seed in "${SEEDS[@]}"; do
    tag="V${V}_L${L}_d${D}_s${seed}"
    echo "[$(date '+%F %T')] start $tag"
    python -u scripts/train.py \
        model_args.vocab_size="$V" \
        model_args.seq_len="$L" \
        model_args.d_model="$D" \
        extra_args.alpha_steps="$ALPHA_STEPS" \
        extra_args.stop_at_accuracy="$STOP_ACC" \
        extra_args.n_prints="$N_PRINTS" \
        extra_args.track_artifacts=false \
        extra_args.experiment_name="$EXP" \
        extra_args.seed="$seed" \
        optim_args.alpha_lr="$ALPHA_LR" \
        model_args.mask1="$MASK_1" \
        data_args.alpha_batch="$ALPHA_BATCH" \
        data_args.gen_device="$GEN_DEVICE" \
        extra_args.matmul_precision="$MATMUL_PRECISION" \
        > "$LOG_DIR/$tag.log" 2>&1 &
    pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[$(date '+%F %T')] done  seed ${SEEDS[$i]}"
    else
        echo "[$(date '+%F %T')] FAIL  seed ${SEEDS[$i]}  (see $LOG_DIR/)"
        status=1
    fi
done

grep -H 'early stop at step' "$LOG_DIR"/V${V}_L${L}_d${D}_s*.log || \
    echo "no run reached $STOP_ACC within the budget -- raise ALPHA_STEPS"

echo "finished   : $(date '+%F %T')  exit=$status"
exit "$status"
