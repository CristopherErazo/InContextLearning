#!/usr/bin/env bash
# Does TF32 shift the measured learning time T*?
#
# TF32 lifts the A100's float32 matmul ceiling from 19.5 to 156 TFLOPS by rounding
# each matmul input's mantissa from 24 bits to 11 (see icl.set_matmul_precision).
# That is a numerical change, and this repo is unusually exposed to it: most
# parameters stay frozen at random init, so the learned signal is small against a
# large random background, and T* is a transition *time* rather than a final value.
#
# The axis that matters is d_model. TF32's error accumulates with the length of the
# reduction, which is d, and d is one of the sweep axes whose exponent is being
# measured -- so a precision artefact would masquerade as a scaling result. Seed
# noise is not the worry; a systematic drift with d is.
#
# Every other knob is held at the workstation sweep's values, and both precisions
# see the same seed, hence the same initial weights and the same batches.
#
#   bash shell/tf32_ab.sh                  # run the grid (sequential; the GPU saturates)
#   bash shell/tf32_ab.sh --report         # parse existing logs, print the comparison
#   DRY_RUN=1 bash shell/tf32_ab.sh        # print the plan
#   DIMS="512" SEEDS="1" bash shell/tf32_ab.sh    # a subset
#
# Run from the repo root.

set -euo pipefail

EXP="${EXP:-tf32_ab}"
SEEDS=(${SEEDS:-1 2 3})
DIMS=(${DIMS:-64 128 512 1024})
PRECISIONS=(${PRECISIONS:-highest high})
BASE_V="${BASE_V:-128}"; BASE_L="${BASE_L:-128}"
ALPHA_STEPS="${ALPHA_STEPS:-20}"
STOP_ACC="${STOP_ACC:-0.75}"
N_PRINTS="${N_PRINTS:-300}"
ALPHA_LR="${ALPHA_LR:-4000}"
ALPHA_BATCH="${ALPHA_BATCH:-1500}"
MASK_1="${MASK_1:-prev}"
GEN_DEVICE="${GEN_DEVICE:-cpu}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a PY_CMD <<< "${PY:-uv run --no-sync python}"

LOG_DIR="logs/$EXP"

# ---------------------------------------------------------------- report
report() {
    python3 - "$LOG_DIR" <<'PY'
import glob, os, re, sys, statistics
log_dir = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(log_dir, "d*_s*.log"))):
    m = re.match(r"d(\d+)_(\w+?)_s(\d+)\.log$", os.path.basename(f))
    if not m: continue
    d, prec, seed = int(m.group(1)), m.group(2), int(m.group(3))
    text = open(f).read()
    stop = re.search(r"early stop at step (\d+)", text)
    rate = re.search(r"\(([\d.]+) ms/step\)", text)
    rows[(d, prec, seed)] = (int(stop.group(1)) if stop else None,
                             float(rate.group(1)) if rate else None)
if not rows:
    print(f"no logs in {log_dir}/"); sys.exit(0)
dims = sorted({k[0] for k in rows}); precs = sorted({k[1] for k in rows}, reverse=True)
print(f"{'d_model':>8} {'precision':>10} {'T* per seed':>26} {'mean T*':>9} {'ms/step':>8}")
print("-" * 68)
summary = {}
for d in dims:
    for p in precs:
        ts = [rows[k][0] for k in sorted(rows) if k[0] == d and k[1] == p and rows[k][0]]
        ms = [rows[k][1] for k in sorted(rows) if k[0] == d and k[1] == p and rows[k][1]]
        if not ts: continue
        summary[(d, p)] = (statistics.mean(ts), statistics.pstdev(ts), ms)
        print(f"{d:8d} {p:>10} {str(ts):>26} {statistics.mean(ts):9.0f} "
              f"{statistics.mean(ms) if ms else float('nan'):8.2f}")
print()
print(f"{'d_model':>8} {'dT*/T*':>9} {'vs seed scatter':>17} {'speedup':>8}   verdict")
print("-" * 68)
for d in dims:
    a, b = summary.get((d, "highest")), summary.get((d, "high"))
    if not (a and b): continue
    rel = (b[0] - a[0]) / a[0]
    scatter = max(a[1], b[1]) / a[0] if a[0] else float("inf")
    speed = (statistics.mean(a[2]) / statistics.mean(b[2])) if a[2] and b[2] else float("nan")
    verdict = "within seed noise" if abs(rel) <= max(scatter, 0.02) else "SHIFTED"
    print(f"{d:8d} {rel:+8.1%} {scatter:16.1%} {speed:7.2f}x   {verdict}")
print("\nA drift in dT*/T* that grows with d_model is the failure mode; a constant")
print("offset inside seed scatter is not. Compare the column, not single rows.")
PY
}

if [[ "${1:-}" == "--report" ]]; then report; exit 0; fi

# ---------------------------------------------------------------- run
mkdir -p "$LOG_DIR"
n=$(( ${#DIMS[@]} * ${#PRECISIONS[@]} * ${#SEEDS[@]} ))
echo "experiment : $EXP   runs: $n (sequential)"
echo "grid       : d=${DIMS[*]}  precision=${PRECISIONS[*]}  seeds=${SEEDS[*]}"
echo "held fixed : V=$BASE_V L=$BASE_L mask1=$MASK_1 alpha_lr=$ALPHA_LR alpha_batch=$ALPHA_BATCH"
echo

i=0
for d in "${DIMS[@]}"; do
  for prec in "${PRECISIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      i=$((i + 1))
      tag="d${d}_${prec}_s${seed}"
      args=(
        -u scripts/train.py
        model_args.vocab_size="$BASE_V" model_args.seq_len="$BASE_L" model_args.d_model="$d"
        model_args.mask1="$MASK_1"
        optim_args.alpha_lr="$ALPHA_LR"
        data_args.alpha_batch="$ALPHA_BATCH" data_args.gen_device="$GEN_DEVICE"
        extra_args.alpha_steps="$ALPHA_STEPS" extra_args.stop_at_accuracy="$STOP_ACC"
        extra_args.n_prints="$N_PRINTS" extra_args.track_artifacts=false
        extra_args.matmul_precision="$prec"
        extra_args.experiment_name="$EXP" extra_args.seed="$seed"
      )
      if [[ "$DRY_RUN" == "1" ]]; then echo "${PY_CMD[*]} ${args[*]}"; continue; fi
      printf '[%s] %2d/%d %-22s ' "$(date '+%H:%M:%S')" "$i" "$n" "$tag"
      if "${PY_CMD[@]}" "${args[@]}" > "$LOG_DIR/$tag.log" 2>&1; then
        echo "$(grep -oE 'early stop at step [0-9]+|training done at step [0-9]+' "$LOG_DIR/$tag.log" | tail -1)"
      else
        echo "FAILED (see $LOG_DIR/$tag.log)"
      fi
    done
  done
done

echo; report
