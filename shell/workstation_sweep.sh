#!/usr/bin/env bash
# L sweep at several embedding dimensions, including d = inf, on a single workstation GPU.
#
# Uses the reduced backend (icl.reduced): SGD in the (M, Q, G) coordinates, exact for
# this model, with a step cost independent of d. Finite d is drawn from the full
# model's init law (init=sample, needs d >= max(L, V)); d=inf is the exact limit
# (every Gram matrix = I). See size_scaling.md. BACKEND=full runs the original
# MinimalTransformer instead (d x d weights; finite d only).
#
# Every run stops at the first evaluation with top1_accuracy >= STOP_ACC; that step is
# the measured learning time T*.
#
#   bash shell/workstation_sweep.sh              # run it
#   DRY_RUN=1 bash shell/workstation_sweep.sh    # print the plan and exit
#   SEEDS="1 2 3" bash shell/workstation_sweep.sh
#   DS="inf" LS="64 128" bash shell/workstation_sweep.sh   # a sub-grid
#   BACKEND=full EXP=L_sweep_full DS=2048 LS="64 128 256 512" bash shell/workstation_sweep.sh
#
# Run from the repo root.

# nohup bash shell/workstation_sweep.sh > ./logs/L_sweep_reduced.log 2>&1 &

set -euo pipefail

# ---------------------------------------------------------------- knobs
EXP="${EXP:-L_sweep_reduced_Vlarge}"           # TrackLab experiment; all runs land together
SEEDS=(${SEEDS:-1})                     # >= 3 seeds per configuration for the fits
LS=(${LS:-64 128 256 512})         # sequence lengths
DS=(${DS:-2048 4096 inf})               # embedding dimensions; "inf" = the d -> inf limit
V="${V:-256}"                           # vocabulary size (fixed in this sweep)
BACKEND="${BACKEND:-reduced}"           # "reduced" (icl.reduced) or "full" (MinimalTransformer)
INIT="${INIT:-sample}"                  # reduced only: "sample" (any d, inf) or "full" (full model's exact draw)
JOBS="${JOBS:-1}"                       # concurrent runs; 1 is fastest once the GPU is busy
ALPHA_STEPS="${ALPHA_STEPS:-30}"        # budget = ALPHA_STEPS x predicted T* (icl.config.predicted_learning_time);
                                        # measured T* is ~4x the prediction at L=64, so keep headroom
STOP_ACC="${STOP_ACC:-0.75}"            # early-exit threshold on in-context accuracy
N_PRINTS="${N_PRINTS:-3000}"            # T* resolution = total_steps / N_PRINTS (1% of predicted T*);
                                        # evaluations stop at T*, so a dense schedule stays cheap
N_PRINTS_MODEL="${N_PRINTS_MODEL:-2}"   # artifact snapshots: 2 = the initial model (step 0) and the final one
                                        # (the early-stop step T*, or total_steps if the budget runs out)
ALPHA_LR="${ALPHA_LR:-3500}"            # eta_0 = lr * d for all runs
MASK_1="${MASK_1:-causal}"              # mask for the first attention layer (prev or causal)
BATCH_SIZE="${BATCH_SIZE:-512}"         # fixed batch size (overrides alpha_batch)
TEST_SIZE="${TEST_SIZE:-4096}"          # test sequences; evaluation runs in chunks of BATCH_SIZE
GEN_DEVICE="${GEN_DEVICE:-auto}"        # batch sampling device: "cpu", "cuda", "auto" (= training device)
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"  # "highest" = true fp32, "high" = TF32
DRY_RUN="${DRY_RUN:-0}"
read -r -a PY_CMD <<< "${PY:-uv run python}"

# Measured on the A100 (reduced backend, V=128, B=512, TF32), per training step:
#   L=128  4.7 ms  0.3 GiB | L=256 9.7 ms 0.6 GiB | L=512 17 ms 1.8 GiB | L=1024 50 ms 5.5 GiB
# The same at every d, so the d=2048, 4096 and inf rows cost the same.

n_runs=$(( ${#DS[@]} * ${#LS[@]} * ${#SEEDS[@]} ))
echo "experiment : $EXP"
echo "grid       : d in (${DS[*]}) x L in (${LS[*]}) at V=$V   seeds: ${SEEDS[*]}   runs: $n_runs   concurrency: $JOBS"
echo "budget     : alpha_steps=$ALPHA_STEPS   stop_at_accuracy=$STOP_ACC   n_prints=$N_PRINTS   n_prints_model=$N_PRINTS_MODEL"
echo "data       : batch_size=$BATCH_SIZE   test_size=$TEST_SIZE   gen_device=$GEN_DEVICE"
echo "model      : backend=$BACKEND$([[ "$BACKEND" == reduced ]] && echo "   init=$INIT")"
echo "numerics   : matmul_precision=$MATMUL_PRECISION   alpha_lr=$ALPHA_LR   mask1=$MASK_1"
echo

# Fail before launching anything rather than midway: d = inf exists only in the reduced
# backend with init=sample, and init=sample needs d >= max(L, V).
if [[ "$BACKEND" != reduced && "$BACKEND" != full ]]; then
    echo "BACKEND must be 'reduced' or 'full', got '$BACKEND'" >&2; exit 1
fi
for D in "${DS[@]}"; do
    if [[ "$D" == inf ]]; then
        if [[ "$BACKEND" != reduced || "$INIT" != sample ]]; then
            echo "d=inf needs BACKEND=reduced INIT=sample" >&2; exit 1
        fi
        continue
    fi
    [[ "$BACKEND" == reduced && "$INIT" == sample ]] || continue
    for L in "${LS[@]}"; do
        if (( D < L || D < V )); then
            echo "d=$D < max(L=$L, V=$V): init=sample cannot draw it (use backend=full)" >&2; exit 1
        fi
    done
done

LOG_DIR="logs/$EXP"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------- run
launch() {
    local L=$1 D=$2 seed=$3
    local tag="V${V}_L${L}_d${D}_s${seed}"
    local model_args_=(model_args.backend="$BACKEND")
    [[ "$BACKEND" == reduced ]] && model_args_+=(model_args.init="$INIT")
    if [[ "$D" == inf ]]; then
        model_args_+=(model_args.infinite_d=true)
    else
        model_args_+=(model_args.d_model="$D")
    fi
    local args=(
        -u scripts/train.py
        "${model_args_[@]}"
        model_args.vocab_size="$V"
        model_args.seq_len="$L"
        model_args.mask1="$MASK_1"
        data_args.batch_size="$BATCH_SIZE"
        data_args.test_size="$TEST_SIZE"
        data_args.gen_device="$GEN_DEVICE"
        optim_args.alpha_lr="$ALPHA_LR"
        extra_args.alpha_steps="$ALPHA_STEPS"
        extra_args.stop_at_accuracy="$STOP_ACC"
        extra_args.n_prints="$N_PRINTS"
        extra_args.n_prints_model="$N_PRINTS_MODEL"
        extra_args.matmul_precision="$MATMUL_PRECISION"
        extra_args.experiment_name="$EXP"
        extra_args.seed="$seed"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "${PY_CMD[*]} ${args[*]}"
        return
    fi
    echo "[$(date '+%F %T')] start $tag"
    "${PY_CMD[@]}" "${args[@]}" > "$LOG_DIR/$tag.log" 2>&1 \
        && echo "[$(date '+%F %T')] done  $tag" \
        || echo "[$(date '+%F %T')] FAIL  $tag  (see $LOG_DIR/$tag.log)"
}

for seed in "${SEEDS[@]}"; do
    for D in "${DS[@]}"; do
        for L in "${LS[@]}"; do
            if [[ "$DRY_RUN" == "1" || "$JOBS" -le 1 ]]; then
                launch "$L" "$D" "$seed"
            else
                # keep at most $JOBS runs in flight
                while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
                launch "$L" "$D" "$seed" &
            fi
        done
    done
done
wait

echo
echo "sweep finished: $(date '+%F %T')"
echo "runs in data/$EXP/ (each run's config.json has seq_len / d_model / infinite_d); per-run stdout in $LOG_DIR/"
echo "T* = the step on the 'early stop at step N' line of each run's info.log; runs that end with"
echo "'training done' never reached STOP_ACC within the budget (raise ALPHA_STEPS):"
echo "  grep -H 'early stop at step\|training done' data/$EXP/run_*/logs/info.log"
