#!/usr/bin/env bash
# Scaling sweep on a single workstation GPU: a consistency check for the same grid
# that shell/leonardo_sweep.sh runs on the cluster. Same configurations, same seeds,
# same per-run overrides -- only the scheduling differs (a plain loop here, a SLURM
# job array there), so the two are directly comparable.
#
# Every run stops at the first evaluation with top1_accuracy >= STOP_ACC; that step is
# the measured learning time T*.
#
#   bash shell/workstation_sweep.sh              # run it
#   DRY_RUN=1 bash shell/workstation_sweep.sh    # print the plan and exit
#   JOBS=3 bash shell/workstation_sweep.sh       # 3 runs at a time on the GPU
#   SWEEPS="L" bash shell/workstation_sweep.sh   # only the L sweep (any of L V d, plus base)
#
# Run from the repo root.

# SWEEPS="base L" EXP="sweep_L1_constrained" DRY_RUN=1 nohup bash shell/workstation_sweep.sh > ./logs/sweep.log 2>&1 &

set -euo pipefail

# ---------------------------------------------------------------- knobs
EXP="${EXP:-scaling_sweep}"          # TrackLab experiment; all runs land together
SEEDS=(${SEEDS:-1 2})              # >= 3 seeds per configuration
JOBS="${JOBS:-1}"                    # concurrent runs; the model is tiny, one GPU fits several
ALPHA_STEPS="${ALPHA_STEPS:-30}"     # budget = ALPHA_STEPS x predicted T* (icl.config.predicted_learning_time)
STOP_ACC="${STOP_ACC:-0.75}"         # early-exit threshold on in-context accuracy
N_PRINTS="${N_PRINTS:-400}"          # T* resolution = total_steps / N_PRINTS (~0.3% of budget)
SWEEPS="${SWEEPS:-base L V d}"       # which sub-sweeps to include
ALPHA_LR="${ALPHA_LR:-3500}"        # learning rate for all runs; the grid is in the sweep axes
MASK_1="${MASK_1:-prev}"             # mask for the first attention layer (prev or causal)
ALPHA_BATCH="${ALPHA_BATCH:-1500}"         # batch size for all runs; the grid is in the sweep axes
BATCH_SIZE="${BATCH_SIZE:-512}"         # override alpha_batch 
# Measured on the A100 workstation 2026-09-21, at the baseline V=128 L=128 d=512:
#   tf32 off, gen cpu   5.00 ms/step      tf32 on, gen cpu    5.00 -> jobs=2: 3.80 agg
#   tf32 on,  gen cuda  3.70 ms/step      tf32 on, gen cuda -> jobs=2: 4.05 agg
# TF32 makes the GPU step so cheap that CPU sampling becomes the critical path, so
# generation moves back onto the GPU and concurrency stops paying. JOBS=1 is best or
# tied-best at every d measured. See shell/tf32_ab.sh for the T* validation of TF32.
GEN_DEVICE="${GEN_DEVICE:-cuda}"     # batch sampling device: "cpu", "cuda", "auto" (= training device)
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"  # "highest" = true fp32, "high" = TF32 (2.2-3.8x at d>=512)
DRY_RUN="${DRY_RUN:-0}"             
read -r -a PY_CMD <<< "${PY:-uv run python}"

# ---------------------------------------------------------------- the grid
# Baseline is V=128 L=128 d=256; each sub-sweep varies one axis and omits the
# baseline point, which is run once below. Keep this identical to leonardo_sweep.sh.
BASE_V=128; BASE_L=128; BASE_D=512
SWEEP_L=(64 256 512 1024)
SWEEP_V=(32 64 256 512)
SWEEP_D=(64 128 512 1024)

# SWEEP_L=(512 1024)
# SWEEP_V=(512 1024)
# SWEEP_D=(1024 2048)

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
for s in $SWEEPS; do
    case "$s" in
        base) add_config "$BASE_V $BASE_L $BASE_D" ;;
        L)    for L in "${SWEEP_L[@]}"; do add_config "$BASE_V $L $BASE_D"; done ;;
        V)    for V in "${SWEEP_V[@]}"; do add_config "$V $BASE_L $BASE_D"; done ;;
        d)    for D in "${SWEEP_D[@]}"; do add_config "$BASE_V $BASE_L $D"; done ;;
        *)    echo "unknown sweep '$s' (expected: base L V d)" >&2; exit 1 ;;
    esac
done

n_runs=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
echo "experiment : $EXP"
echo "configs    : ${#CONFIGS[@]}   seeds: ${SEEDS[*]}   runs: $n_runs   concurrency: $JOBS"
echo "budget     : alpha_steps=$ALPHA_STEPS   stop_at_accuracy=$STOP_ACC   n_prints=$N_PRINTS"
echo "numerics   : matmul_precision=$MATMUL_PRECISION   gen_device=$GEN_DEVICE   mask1=$MASK_1"
echo

LOG_DIR="logs/$EXP"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------- run
launch() {
    local V=$1 L=$2 D=$3 seed=$4
    local tag="V${V}_L${L}_d${D}_s${seed}"
    local args=(
        -u scripts/train.py
        model_args.vocab_size="$V"
        model_args.seq_len="$L"
        model_args.d_model="$D"
        extra_args.alpha_steps="$ALPHA_STEPS"
        extra_args.stop_at_accuracy="$STOP_ACC"
        extra_args.n_prints="$N_PRINTS"
        extra_args.track_artifacts=false
        extra_args.experiment_name="$EXP"
        extra_args.seed="$seed"
        optim_args.alpha_lr="$ALPHA_LR"
        model_args.mask1="$MASK_1"
        data_args.alpha_batch="$ALPHA_BATCH"
        data_args.gen_device="$GEN_DEVICE"
        extra_args.matmul_precision="$MATMUL_PRECISION"
        data_args.batch_size="$BATCH_SIZE"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "${PY_CMD[*]} ${args[*]}"
        return
    fi
    echo "[$(date '+%F %T')] start $tag"
    "${PY_CMD[@]}" "${args[@]}" > "$LOG_DIR/$tag.log" 2>&1 \
        && echo "[$(date '+%F %T')] done  $tag  -> $(grep -c . "$LOG_DIR/$tag.log") log lines" \
        || echo "[$(date '+%F %T')] FAIL  $tag  (see $LOG_DIR/$tag.log)"
}

for cfg in "${CONFIGS[@]}"; do
    read -r V L D <<< "$cfg"
    for seed in "${SEEDS[@]}"; do
        if [[ "$DRY_RUN" == "1" || "$JOBS" -le 1 ]]; then
            launch "$V" "$L" "$D" "$seed" "$"
        else
            # keep at most $JOBS runs in flight
            while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
            launch "$V" "$L" "$D" "$seed" &
        fi
    done
done
wait

echo
echo "sweep finished: $(date '+%F %T')"
echo "runs in data/$EXP/ ; per-run logs in $LOG_DIR/"
echo "T* = the step reported on the 'early stop at step N' line of each log:"
echo "  grep -H 'early stop at step' $LOG_DIR/*.log"
