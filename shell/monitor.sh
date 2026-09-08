#!/usr/bin/env bash

THRESHOLD="${1:-0.5}"
INTERVAL="${2:-30}"
BASE_DIR="${BASE_DIR:-$PWD/data/swap_experiments}"  # Default base directory for runs
NO_RUN_TIMEOUT="${NO_RUN_TIMEOUT:-1800}"   # 30 minutes without new run

last_seen_run=""
last_activity_time=$(date +%s)

latest_run() {
    find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -name 'run_[0-9]*' 2>/dev/null \
        | sort -V \
        | tail -n 1
}

while true; do
    run_dir="$(latest_run)"

    if [[ -n "$run_dir" ]]; then
        run_name="$(basename "$run_dir")"
        log_file="$run_dir/logs/info.log"

        if [[ -f "$log_file" ]]; then
            line="$(grep 'top1_accuracy=' "$log_file" | tail -n 1)"
            if [[ -n "$line" ]]; then
                acc="$(printf '%s\n' "$line" | sed -E 's/.*top1_accuracy=([0-9.]+).*/\1/')"

                printf '%s | %s | accuracy=%s\n' "$(date '+%F %T')" "$run_name" "$acc"

                if awk "BEGIN { exit !($acc > $THRESHOLD) }"; then
                    echo "Threshold reached: $acc > $THRESHOLD"
                    pid="$(pgrep -af 'python.*scripts/launcher.py' | tail -n 1 | awk '{print $1}')"
                    if [[ -n "$pid" ]]; then
                        kill "$pid"
                        echo "Stopped current run PID $pid; continuing to next run"
                    else
                        echo "No active launcher process found for the current run."
                    fi
                    echo "Monitoring continues for the next run..."
                fi

                last_activity_time=$(date +%s)
                last_seen_run="$run_name"
            fi
        fi
    else
        now=$(date +%s)
        age=$((now - last_activity_time))
        if (( age > NO_RUN_TIMEOUT )); then
            echo "No new runs for ${NO_RUN_TIMEOUT}s; exiting monitor."
            exit 0
        fi
        echo "No run directory yet... waiting"
    fi

    sleep "$INTERVAL"
done