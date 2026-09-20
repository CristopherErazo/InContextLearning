#!/usr/bin/env bash
# Calibration scan for eta_0 (= optim_args.alpha_lr, since lr = alpha_lr / d_model).
#
# The theory has no optimum: T* = C * V^2 sqrt(L) / eta_0 falls monotonically, because
# gradient flow has no step size. The optimum is set entirely by what the theory omits --
# discretisation and minibatch noise -- so it has to be measured. Expect slope -1 on a
# log-log plot of T* against eta_0, then a turn-up (noise) and/or a cliff (curvature).
# The slope itself is worth having: -1 is the check that one SGD step == one unit of
# gradient-flow time.
#
# Each point gets the same budget *in units of its own predicted T**, so every run costs
# about the same when the theory holds, and the runs that hit the budget are exactly the
# ones where it stopped holding. That falls out of extra_args.alpha_steps: the prediction
# in icl.config.predicted_learning_time carries the 1/eta_0, so the budget tracks the grid
# automatically.
#
#   bash shell/eta_scan.sh                       # baseline V=128 L=128 d=256
#   DRY_RUN=1 bash shell/eta_scan.sh             # print the plan
#   V=512 bash shell/eta_scan.sh                 # corner check at large V
#   L=512 SEEDS=1 bash shell/eta_scan.sh         # corner check at large L, one seed
#
# Run from the repo root.

set -euo pipefail

# ---------------------------------------------------------------- knobs
EXP="${EXP:-eta_scan}"
V="${V:-128}"; L="${L:-128}"; D="${D:-256}"
ALPHA_LRS=(${ALPHA_LRS:-625 1250 2500 5000 10000 20000 40000})   # eta_0, x2 around the default
SEEDS=(${SEEDS:-1 2 3})
BUDGET="${BUDGET:-4}"        # total_steps = BUDGET x predicted T*, per point
STOP_ACC="${STOP_ACC:-0.75}"
N_PRINTS="${N_PRINTS:-300}"
JOBS="${JOBS:-2}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a PY_CMD <<< "${PY:-uv run python}"

LOG_DIR="logs/$EXP"
mkdir -p "$LOG_DIR"

echo "experiment : $EXP    config: V=$V L=$L d=$D"
echo "eta_0 grid : ${ALPHA_LRS[*]}"
echo "seeds      : ${SEEDS[*]}    budget: ${BUDGET}x predicted T*    concurrency: $JOBS"
echo

# ---------------------------------------------------------------- run
launch() {
    local alr=$1 seed=$2
    local tag="V${V}_L${L}_d${D}_alr${alr}_s${seed}"   # config in the name: eta_corners.sh reuses one EXP
    local args=(
        -u scripts/train.py
        model_args.vocab_size="$V"
        model_args.seq_len="$L"
        model_args.d_model="$D"
        optim_args.alpha_lr="$alr"
        extra_args.alpha_steps="$BUDGET"
        extra_args.stop_at_accuracy="$STOP_ACC"
        extra_args.n_prints="$N_PRINTS"
        extra_args.track_artifacts=false
        extra_args.experiment_name="$EXP"
        extra_args.seed="$seed"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "${PY_CMD[*]} ${args[*]}"
        return
    fi
    echo "[$(date '+%F %T')] start $tag"
    "${PY_CMD[@]}" "${args[@]}" > "$LOG_DIR/$tag.log" 2>&1 || true
}

for alr in "${ALPHA_LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [[ "$DRY_RUN" == "1" || "$JOBS" -le 1 ]]; then
            launch "$alr" "$seed"
        else
            while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
            launch "$alr" "$seed" &
        fi
    done
done
wait

[[ "$DRY_RUN" == "1" ]] && exit 0

# ---------------------------------------------------------------- summary
# Three outcomes, kept distinct: converged (T* = the step), hit the budget without
# reaching STOP_ACC (noise regime), or diverged (curvature cliff).
echo
printf '%8s  %10s  %s\n' "eta_0" "T* (steps)" "per-seed outcome"
for alr in "${ALPHA_LRS[@]}"; do
    outcomes=(); sum=0; n=0
    for seed in "${SEEDS[@]}"; do
        f="$LOG_DIR/V${V}_L${L}_d${D}_alr${alr}_s${seed}.log"
        if [[ ! -f "$f" ]]; then
            outcomes+=("missing")
        elif grep -q 'diverged at step' "$f"; then
            s=$(grep 'diverged at step' "$f" | tail -1 | sed -E 's/.*diverged at step ([0-9]+).*/\1/')
            outcomes+=("nan@$s")
        elif grep -q 'early stop at step' "$f"; then
            t=$(grep 'early stop at step' "$f" | tail -1 | sed -E 's/.*early stop at step ([0-9]+).*/\1/')
            outcomes+=("$t"); sum=$((sum + t)); n=$((n + 1))
        elif grep -q 'training done at step' "$f"; then
            outcomes+=("budget")
        else
            outcomes+=("crash")
        fi
    done
    mean=$([[ $n -gt 0 ]] && echo $((sum / n)) || echo "-")
    printf '%8s  %10s  %s\n' "$alr" "$mean" "${outcomes[*]}"
done

echo
echo "Read it as: T* should fall as 1/eta_0 (slope -1 in log-log) while the runs are clean."
echo "'budget' marks where noise stops the escape; 'nan' marks the discretisation cliff."
echo "Pick eta_0 a factor 2-3 BELOW the minimum, and use that one value for the whole"
echo "sweep -- tuning it per configuration would leak straight into the fitted exponent."
