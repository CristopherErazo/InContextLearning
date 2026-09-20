#!/usr/bin/env bash
# Pick ONE eta_0 for the whole Leonardo sweep. Wraps shell/eta_scan.sh and repeats it at
# the cheap end of each axis, then prints a config x eta_0 matrix of T*.
#
# Why the small end: cost is T* ~ V^2 sqrt(L), so small V and small L are the fast configs
# -- and since lr = alpha_lr / d_model, small d is where the *actual* learning rate is
# largest and instability appears first. The cheap corners are also the dangerous ones.
#
#   bash shell/eta_corners.sh                      # the matrix
#   DRY_RUN=1 bash shell/eta_corners.sh            # print the plan
#   SEEDS="1 2" JOBS=3 bash shell/eta_corners.sh   # cheaper / more parallel
#
# Run from the repo root.

set -euo pipefail

EXP="${EXP:-eta_corners}"
ALPHA_LRS="${ALPHA_LRS:-1250 2500 5000 10000 20000}"   # one below the default, four at/above 1250 2500 5000 10000
SEEDS="${SEEDS:-2 3}"
STOP_ACC="${STOP_ACC:-0.75}"   # forwarded to eta_scan.sh so both agree
BUDGET="${BUDGET:-4}"          # total_steps = BUDGET x predicted T*
JOBS="${JOBS:-2}"
DRY_RUN="${DRY_RUN:-0}"

# baseline, then the small end of each sweep axis (baseline = V128 L128 d256).
# Override with a ';'-separated list: CONFIGS="128 128 256;64 128 256"
IFS=';' read -r -a CONFIGS <<< "${CONFIGS:-128 128 256;128 32 256;128 64 256;64 128 256;128 128 64;128 128 128}"

echo "eta_0 grid : $ALPHA_LRS"
echo "configs    : ${#CONFIGS[@]}   seeds: $SEEDS   budget: ${BUDGET}x predicted T*"
echo

for cfg in "${CONFIGS[@]}"; do
    read -r V L D <<< "$cfg"
    echo "### V=$V L=$L d=$D"
    EXP="${EXP}" V="$V" L="$L" D="$D" \
    ALPHA_LRS="$ALPHA_LRS" SEEDS="$SEEDS" BUDGET="$BUDGET" JOBS="$JOBS" STOP_ACC="$STOP_ACC" \
    DRY_RUN="$DRY_RUN" PY="${PY:-uv run python}" \
        bash shell/eta_scan.sh
    echo
done

[[ "$DRY_RUN" == "1" ]] && exit 0

cat <<EOF
All runs are in data/$EXP/ , one run_NNN per (config, eta_0, seed).
Each carries config.json (vocab_size, seq_len, d_model, alpha_lr, batch_size, seed, ...)
and metrics.jsonl, one JSON object per line in LONG format:
  {"step": 97, "metric": "top1_accuracy", "value": 0.774}
so filter on 'metric' before reading 'value'. Everything needed is there:

  T*          first step whose top1_accuracy >= $STOP_ACC
  diverged    loss is NaN (a bare NaN literal; json/pandas read it, strict parsers do not)
  censored    last step == config.json total_steps and the threshold was never reached

Pick the largest eta_0 that is clean in EVERY config -- one value has to serve the whole
sweep, since tuning it per configuration would leak straight into the fitted exponent --
then take a factor 2-3 below it.

Not covered here: the large-V and large-L corners. Before committing the full sweep,
confirm the chosen value once at the expensive end:
  V=512 SEEDS=1 ALPHA_LRS="<chosen> <2x chosen>" bash shell/eta_scan.sh
EOF
