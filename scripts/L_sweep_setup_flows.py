"""Effective-model flows for the setups of notebooks/26-10-05_Ansatz-check.ipynb, over the
L-sweep of L_sweep_B1024 (d = inf, batch 1024, L = 32 ... 1024), one or more seeds.

For every (seed, L), from the order parameters measured on that run's own step-0 matrices
(the six means, the raw sub-diagonal of M as "M_profile", the eight block variances),
integrate the gradient flow of the effective loss (method="mean" by default) for each
setup: exactly the keys of the setup move, with `exact_rates` (the d = inf SGD rates:
a mean eta_0 kappa_X / n, the profile eta_0 per entry, a variance 4 eta_0 kappa_X / n
times itself), and every other order parameter is absent, i.e. 0. The profile moves entry
by entry (no family). Note that "M_profile" at step 0 is the raw sub-diagonal, which
already carries the spread of those entries; a setup with both "M_profile" and
"var_M_on" counts that spread twice (drop "var_M_on" from SETUPS to avoid it).

The flow is integrated in chunks (the first 2x the real run's length, then doubling)
until the loss has gone STOP_FRACTION of the way from log V to asymptotic_loss, the stop
of the L_sweep_B1024 runs, or MAX_FACTOR x the real run's length. RK45: with the
511-1023 profile entries LSODA needs 2-3x more loss evaluations for the same flow.

    nohup setsid uv run --no-sync python -u scripts/L_sweep_setup_flows.py > logs/setup_flows.log 2>&1 &
    uv run --no-sync python -u scripts/L_sweep_setup_flows.py --seeds 13 --Ls 64 512 --setups signal_only

Writes data/effective_setups_B1024/flows/<setup>_L<L>_seed<seed>.pkl (the flow DataFrame,
indexed by step, one column per order parameter, "loss") and
data/effective_setups_B1024/learning_times.csv (T* at f = 0.2 and 0.5 of the flow and of
the real run, per (seed, L, setup)). A (seed, L, setup) already in the csv is skipped, so
the script can be restarted.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from icl import PROFILE, EffectiveLoss, RunData, asymptotic_loss, exact_rates, integrate, learning_time

CODE = Path(__file__).resolve().parents[1]
SOURCE = "L_sweep_B1024"
OUTPUT = CODE / "data" / "effective_setups_B1024"

SETUPS = {"signal_only": ["M_on", "Q_on", "G_on"]}
SETUPS["means_only"] = SETUPS["signal_only"] + ["M_off", "Q_T", "G_T"]
SETUPS["means_&_noise"] = SETUPS["means_only"] + ["var_M_on", "var_M_off", "var_Q_on", "var_Q_T", "var_Q_N",
                                                  "var_G_on", "var_G_T", "var_G_N"]
SETUPS["means_&_profile"] = SETUPS["means_only"] + ["M_profile"]
SETUPS["means,_noise_&_profile"] = SETUPS["means_&_noise"] + ["M_profile"]

FRACTIONS = (0.2, 0.5)
STOP_FRACTION, MAX_FACTOR, POINTS_PER_CHUNK, NUM_MUS = 0.5, 50, 200, 64


def source_runs(seeds, Ls) -> dict:
    """{(seed, L): RunData} of the d = inf runs of SOURCE."""
    runs = {}
    for run_dir in sorted((CODE / "data" / SOURCE).glob("run_*")):
        config = json.loads((run_dir / "config.json").read_text())
        seed, L = config["extra_args"]["seed"], config["model_args"]["seq_len"]
        if config["model_args"]["infinite_d"] and seed in seeds and (not Ls or L in Ls):
            runs[(seed, L)] = RunData(SOURCE, run_dir.name, base_dir=CODE / "data")
    return runs


def flow_until_learned(loss, start: dict, rates: dict, V: int, L: int, K: int, first_chunk: float,
                       max_steps: float) -> pd.DataFrame:
    """`integrate` in chunks, each restarting from the last state of the previous one, until the
    loss has gone STOP_FRACTION of the way to asymptotic_loss or `max_steps`."""
    target_loss = math.log(V) - STOP_FRACTION * (math.log(V) - asymptotic_loss(V, L, K))
    pieces, state, offset, chunk = [], dict(start), 0.0, first_chunk
    while offset < max_steps:
        length = min(chunk, max_steps - offset)
        flow = integrate(loss, state, rates, steps=length, record_steps=np.linspace(0, length, POINTS_PER_CHUNK + 1),
                         method="RK45")
        flow.index = flow.index + offset
        pieces.append(flow if not pieces else flow.iloc[1:])
        if flow["loss"].min() <= target_loss or flow.index[-1] < offset + length - 1e-6:    # learned, or solver stopped
            break
        last = flow.iloc[-1]
        state = {name: torch.as_tensor(last[name]) if name == PROFILE else float(last[name]) for name in start}
        offset += length
        chunk *= 2
    return pd.concat(pieces)


def main(seeds, Ls, setups, method, device):
    (OUTPUT / "flows").mkdir(parents=True, exist_ok=True)
    csv = OUTPUT / "learning_times.csv"
    table = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    done = set(zip(table["seed"], table["L"], table["setup"])) if len(table) else set()
    for (seed, L), run in sorted(source_runs(seeds, Ls).items(), key=lambda item: (item[0][0], item[0][1])):
        config = run.config
        V, K, beta = config.model_args.vocab_size, config.data_args.K, config.model_args.beta
        real_loss = run.metrics["loss"].dropna()
        real_length = int(real_loss.index[-1])
        real_T = {f: learning_time(real_loss, V, L, K, f) for f in FRACTIONS}
        initial = run.order_params(0, profile=True, variances=True)
        mus = np.unique(np.linspace(1, L, NUM_MUS).round().astype(int))
        loss = EffectiveLoss(V, L, K, beta, method=method, mus=mus, device=device)
        for setup in setups:
            if (seed, L, setup) in done:
                continue
            keys = SETUPS[setup]
            calls = [0]

            def counted(order_params):
                calls[0] += 1
                return loss(order_params)

            clock = time.perf_counter()
            flow = flow_until_learned(counted, {name: initial[name] for name in keys}, exact_rates(config, keys),
                                      V, L, K, first_chunk=2 * real_length, max_steps=MAX_FACTOR * real_length)
            seconds = time.perf_counter() - clock
            pd.to_pickle(flow, OUTPUT / "flows" / f"{setup}_L{L}_seed{seed}.pkl")
            T = {f: learning_time(flow["loss"], V, L, K, f) for f in FRACTIONS}
            row = {"seed": seed, "L": L, "setup": setup, "run": run.run_id, "method": method,
                   **{f"T_{f}": T[f] for f in FRACTIONS}, **{f"T_real_{f}": real_T[f] for f in FRACTIONS},
                   "steps": flow.index[-1], "final_loss": flow["loss"].iloc[-1], "loss_evaluations": calls[0],
                   "seconds": seconds}
            table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
            table.to_csv(csv, index=False)
            print(f"seed={seed} L={L:4d} {setup:24s} T*={T[0.2]:9.1f} (real {real_T[0.2]:8.1f})  "
                  f"{calls[0]:5d} evals, {seconds:6.0f} s", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[13])
    parser.add_argument("--Ls", type=int, nargs="*", default=[], help="default: every L of the sweep")
    parser.add_argument("--setups", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--method", default="mean", choices=["mean", "mc"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(set(args.seeds), set(args.Ls), args.setups, args.method, args.device)
