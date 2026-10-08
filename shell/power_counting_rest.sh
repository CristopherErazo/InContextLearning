#!/usr/bin/env bash
# The rest of shell/power_counting_runs.sh (V=128 L=1024 seed 2; V=256 L=512 seeds 42, 2), each
# stopped once it is AFTER steps past t_mid (the first evaluation with accuracy >= 1/2): the
# power counting needs t - t_mid <= 1555 (run_001's last snapshot), and the budgets in
# total_steps are generous guesses of t_mid. A run that stops this way keeps its configured
# total_steps in config.json; its last logged step is the stop. First it attaches to the run
# that is already going (WATCH_DIR / WATCH_PID).
#
#   nohup bash shell/power_counting_rest.sh > logs/power_counting_rest.log 2>&1 &

set -uo pipefail

EXP="${EXP:-power_counting}"
AFTER="${AFTER:-1700}"
PY="${PY:-.venv/bin/python}"

progress() {   # "t_mid last_step" of a run directory (t_mid = -1 before the transition)
  "$PY" - "$1" <<'EOF'
import json, sys
t_mid, last = -1, -1
with open(f"{sys.argv[1]}/metrics.jsonl") as f:
    for line in f:
        row = json.loads(line)
        last = max(last, row["step"])
        if t_mid < 0 and row["metric"] == "top1_accuracy" and row["value"] >= 0.5:
            t_mid = row["step"]
print(t_mid, last)
EOF
}

watch() {      # stop process $2 (run directory $1) once it is AFTER steps past t_mid
  local dir=$1 pid=$2
  while kill -0 "$pid" 2>/dev/null; do
    if [[ -f "$dir/metrics.jsonl" ]]; then
      read -r t_mid last <<< "$(progress "$dir")"
      if (( t_mid >= 0 && last >= t_mid + AFTER )); then
        echo "$(date '+%F %T')  $dir: t_mid=$t_mid, step $last: stopping"
        kill "$pid"; wait "$pid" 2>/dev/null
        return
      fi
    fi
    sleep 20
  done
  echo "$(date '+%F %T')  $dir: finished"
}

if [[ -n "${WATCH_DIR:-}" ]]; then watch "$WATCH_DIR" "$WATCH_PID"; fi

for config in "128 1024 6000 2" "256 512 30000 42" "256 512 30000 2"; do
  read -r V L STEPS SEED <<< "$config"
  before=$(ls -d data/"$EXP"/run_* 2>/dev/null | sort | tail -1)
  echo "$(date '+%F %T')  V=$V L=$L steps=$STEPS seed=$SEED"
  "$PY" -u scripts/train.py model_args.vocab_size="$V" model_args.seq_len="$L" \
       extra_args.total_steps="$STEPS" extra_args.n_prints=$(( STEPS / 10 )) extra_args.n_prints_model=30 \
       extra_args.seed="$SEED" extra_args.experiment_name="$EXP" extra_args.base_dir=./data \
       data_args.test_size=1024 extra_args.log_to_terminal=false &
  pid=$!
  dir=$before
  while [[ "$dir" == "$before" ]]; do sleep 5; dir=$(ls -d data/"$EXP"/run_* | sort | tail -1); done
  watch "$dir" "$pid"
done
echo "$(date '+%F %T')  done"
