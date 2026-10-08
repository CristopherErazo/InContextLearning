#!/usr/bin/env bash
# Runs for the power counting of the pruned effective model (Phase 5 of
# paper/scratch/2026-10-08-1103_plan-pruned-effective-model.md): full_ansatz/run_001's setup
# (reduced backend, init=full, d=4096, B=1024, alpha_lr=3000, beta=0.25, rho=0.2, every order
# parameter logged at every evaluation) with one of V, L changed, two seeds each. They give the
# L and V scaling of the order parameters at matched stages of training.
#
#   nohup bash shell/power_counting_runs.sh > logs/power_counting.log 2>&1 &
#   DRY_RUN=1 bash shell/power_counting_runs.sh
#
# Run from code/. Sequential on one GPU (the cheap runs first).

set -euo pipefail

EXP="${EXP:-power_counting}"
SEEDS=(${SEEDS:-42 2})
DRY_RUN="${DRY_RUN:-0}"
PY="${PY:-.venv/bin/python}"

# V L total_steps: ~2-3 x the expected transition time (run_001: t_mid ~ 1450 at V=128, L=512;
# T* ~ V^2 sqrt(L) by the theory estimate, steeper in V in the L sweeps, hence the V=256 budget)
CONFIGS=(
  "128 128 3000"
  "128 256 3000"
  "64 256 3000"
  "64 512 3000"
  "128 1024 6000"
  "256 512 30000"
)

for config in "${CONFIGS[@]}"; do
  read -r V L STEPS <<< "$config"
  for SEED in "${SEEDS[@]}"; do
    # one evaluation (scalars, order parameters, profile) every 10 steps, as run_001
    N_PRINTS=$(( STEPS / 10 ))
    CMD=("$PY" -u scripts/train.py
         model_args.vocab_size="$V" model_args.seq_len="$L"
         extra_args.total_steps="$STEPS" extra_args.n_prints="$N_PRINTS" extra_args.n_prints_model=30
         extra_args.seed="$SEED" extra_args.experiment_name="$EXP" extra_args.base_dir=./data
         data_args.test_size=1024 extra_args.log_to_terminal=false)
    echo "$(date '+%F %T')  V=$V L=$L steps=$STEPS seed=$SEED"
    if [[ "$DRY_RUN" == 1 ]]; then echo "  ${CMD[*]}"; continue; fi
    "${CMD[@]}"
  done
done
echo "$(date '+%F %T')  done"
