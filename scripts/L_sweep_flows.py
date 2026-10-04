"""Effective-model flows over the d = inf runs of L_sweep_reduced.

For every run, from the extended-ansatz order parameters of its own initial
draw (step 0), integrate the gradient flow of the effective loss (method="mean")
for three nested variants of the ansatz:

    signal    M_on, Q_on, G_on
    trigger   + Q_T, G_T
    extended  + M_off

with the rates alpha_lr / (support size), exact at d = inf on the ansatz. Its
learning time is `learning_time(flow["loss"], ...)`: the step at which the loss
has gone FRACTION of the way from log V to the manuscript's L^infty.

    uv run python -u scripts/L_sweep_flows.py [run_011 run_026 ...]
    uv run python -u scripts/L_sweep_flows.py --source L_sweep_reduced_sigma0.5_loss_stop

Writes data/effective_L_sweep/flows.pkl ({(run, variant): DataFrame}) and
data/effective_L_sweep/learning_times.csv (data/effective_<source>/ for another
--source). The control runs are
scripts/L_sweep_controls.py; notebooks/26-10-04_L_sweep_effective.ipynb compares.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from icl import EffectiveLoss, RunData, integrate, learning_time, measure_order_params, support_sizes

CODE = Path(__file__).resolve().parents[1]
VARIANTS = {"signal": ("M_on", "Q_on", "G_on"),
            "trigger": ("M_on", "Q_on", "G_on", "Q_T", "G_T"),
            "extended": ("M_on", "Q_on", "G_on", "Q_T", "G_T", "M_off")}
FRACTION, MAX_STEPS, RECORD_EVERY, NUM_MUS = 0.2, 60000, 10, 64


def initial_order_params(run_dir: Path, K: int) -> dict:
    old_layout = run_dir / "artifacts/matrices/M_step_0.npy"       # runs saved before the ComposedMatrices dict
    if old_layout.exists():
        matrices = {name: np.load(run_dir / f"artifacts/matrices/{name}_step_0.npy") for name in ("M", "Q", "G")}
    else:
        matrices = RunData(run_dir.parent.name, run_dir.name, base_dir=run_dir.parents[1]).matrices(0)
    return measure_order_params(matrices, K)


def main(names, source):
    SOURCE = CODE / "data" / source
    OUTPUT = CODE / "data" / ("effective_L_sweep" if source == "L_sweep_reduced" else f"effective_{source}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    flows_path = OUTPUT / "flows.pkl"
    flows = pickle.loads(flows_path.read_bytes()) if flows_path.exists() else {}
    rows = []
    for run_dir in sorted(SOURCE.glob("run_*")):
        config = json.loads((run_dir / "config.json").read_text())
        if not config["model_args"]["infinite_d"] or (names and run_dir.name not in names):
            continue
        V, L, K = config["model_args"]["vocab_size"], config["model_args"]["seq_len"], config["data_args"]["K"]
        beta, alpha_lr = config["model_args"]["beta"], config["optim_args"]["alpha_lr"]
        start = initial_order_params(run_dir, K)
        sizes = support_sizes(L, V, K)
        mus = np.unique(np.linspace(1, L, NUM_MUS).round().astype(int))
        loss = EffectiveLoss(V, L, K, beta, method="mean", mus=mus)
        for variant, moving in VARIANTS.items():
            clock = time.perf_counter()
            flow = integrate(loss, {name: start[name] for name in moving},
                             {name: alpha_lr / sizes[name] for name in moving},
                             steps=MAX_STEPS, record_steps=np.arange(0, MAX_STEPS + 1, RECORD_EVERY))
            flows[(run_dir.name, variant)] = flow
            T = learning_time(flow["loss"], V, L, K, FRACTION)
            rows.append({"run": run_dir.name, "L": L, "seed": config["extra_args"]["seed"], "variant": variant,
                         "T_star": T, **{f"{name}_0": start[name] for name in moving}})
            print(f"{run_dir.name} L={L:4d} seed={rows[-1]['seed']} {variant:9s} T*={T:8.1f} "
                  f"({time.perf_counter() - clock:.0f} s)", flush=True)
        flows_path.write_bytes(pickle.dumps(flows))
    table = pd.DataFrame(rows)
    csv = OUTPUT / "learning_times.csv"
    if names and csv.exists():                       # keep the other runs' rows
        old = pd.read_csv(csv)
        table = pd.concat([old[~old["run"].isin(table["run"])], table]).sort_values(["run", "variant"])
    table.to_csv(csv, index=False)


if __name__ == "__main__":
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--source", default="L_sweep_reduced")
    args = parser.parse_args()
    main(set(args.runs), args.source)
