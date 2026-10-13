"""Ablation of the logit noise of the effective model on the L-sweep (thread noise-ablation,
journal D31): which noise mechanisms make T*(L) fall with L. Groups and the gradient split are
in `icl.theory.ablation`. Outputs (CSV, JSON) in data/noise_ablation/:

    gradients       the split of d loss / d theta by group (cuts A and B) at points of the
                    L_sweep_B1024 runs (30 seeds x 11 L): t / T*_run in POINT_TIMES and
                    progress f = ESCAPE_FRACTION (early escape); the mass of trigger clusters
                    where a closure moment goes negative when one group is dropped.
                    -> gradient_split.csv, negative_moments.csv
    flows           effective flows of the means_&_noise setup of L_sweep_setup_flows.py
                    (typical init, sign +1, exact_rates) with EffectiveLoss(keep=mode), mode
                    one of "full", "signal", "drop:G[+G...]", "add:G[+G...]", optionally followed by
                    ",zero:var_X[+var_Y...]": those block variances are 0 and frozen (absent from the
                    start and the rates), which removes every term that contains them.
                    -> flows/<mode>_L<L>.csv (trajectory), flows/<mode>_L<L>.json (T*, ...)
    flow-gradients  the split along the full flows, at t / T*_flow in POINT_TIMES and f = ESCAPE_FRACTION.
                    -> flow_gradient_split.csv
    summary         -> gradient_summary.csv, T_star.csv

    uv run --no-sync python -u scripts/noise_ablation.py gradients
    nohup setsid uv run --no-sync python -u scripts/noise_ablation.py flows --modes full signal \\
        > logs/noise_ablation.log 2>&1 &

The SGD runs' order parameters are their logged metrics (the six means, the eight block
variances); with the scalar M_on (no profile) var_M_on is the run's var_M_on_raw, the spread
of the sub-diagonal around its mean (--var-M-on smooth: around the smoothed profile).
Flows stop at f = STOP_FRACTION (so the early-escape point exists), checked after each chunk
(one SGD length, doubling after CONSTANT_CHUNKS), or at MAX_FACTOR x the median length of the
SGD runs (then T* is censored). A flow with its JSON is skipped
(restartable).
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "scripts"))
from L_sweep_setup_flows import SETUPS, real_learning_time, source_runs, typical_order_params  # noqa: E402
from icl import (ORDER_PARAMS, VARIANCES, EffectiveLoss, RunData, asymptotic_loss, exact_rates, integrate,  # noqa: E402
                 learning_time)
from icl.theory.ablation import CUTS, GROUPS, gradient_split, keep_add, keep_drop, negative_moments  # noqa: E402

OUT = CODE / "data" / "noise_ablation"
SOURCE = "L_sweep_B1024"
KEYS = SETUPS["means_&_noise"]                      # the six means and the eight block variances
POINT_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)          # t / T* (T* at f = 0.2)
ESCAPE_FRACTION, FRACTION = 0.35, 0.2
STOP_FRACTION, MAX_FACTOR, POINTS_PER_CHUNK, NUM_MUS = 0.4, 20, 200, 64
CONSTANT_CHUNKS = 4                                 # chunks of one SGD length up to 4 of them, then doubling
NEGATIVE_TIMES = (0.0, 0.5, 1.0)                    # t / T* where a flow's dropped moments are checked


def zeroed_of(mode: str) -> set:
    """The block variances a mode sets to 0 (its ",zero:..." suffix)."""
    if ",zero:" not in mode:
        return set()
    names = set(mode.split(",zero:")[1].split("+"))
    unknown = names - set(VARIANCES)
    if unknown:
        raise ValueError(f"not block variances: {sorted(unknown)}")
    return names


def keep_of(mode: str):
    """The keep of a mode (its ",zero:..." suffix aside). "full" is keep_drop() (every term, the means
    from the term layer, as in every ablation mode); "full_table2" is keep=None (the means from Table 2,
    ansatz_logits), which differs from it at mu <= 2."""
    mode = mode.split(",zero:")[0]
    if mode == "full_table2":
        return None
    if mode == "full":
        return keep_drop()
    if mode == "signal":
        return keep_add()
    kind, names = mode.split(":")
    return {"drop": keep_drop, "add": keep_add}[kind](*names.split("+"))


def make_loss(config, keep=None) -> EffectiveLoss:
    L = config.model_args.seq_len
    mus = np.unique(np.linspace(1, L, NUM_MUS).round().astype(int))
    return EffectiveLoss.from_config(config, method="mean", mus=mus, keep=keep)


def progress_step(loss: pd.Series, config, fraction: float) -> float:
    return learning_time(loss, config.model_args.vocab_size, config.model_args.seq_len, config.data_args.K, fraction)


def point_steps(loss: pd.Series, config) -> list[tuple[str, float]]:
    """[(point, step)]: t / T* in POINT_TIMES and the early-escape fraction."""
    T = progress_step(loss, config, FRACTION)
    points = [(f"t{time_:.2f}", time_ * T) for time_ in POINT_TIMES]
    return points + [(f"f{ESCAPE_FRACTION:.2f}", progress_step(loss, config, ESCAPE_FRACTION))]


def split_rows(loss, order_params: dict) -> list[dict]:
    rows = []
    for cut in CUTS:
        split = gradient_split(loss, order_params, cut)
        parts = split if cut == "A" else {name: split[name] for name in CUTS[cut]}   # total, means, signal once
        rows += [{"cut": cut, "part": part, "param": param, "theta": order_params[param], "value": value}
                 for part, values in parts.items() for param, value in values.items()]
    return rows


# ---- gradients along the SGD runs ------------------------------------------------------

def run_order_params(metrics: pd.DataFrame, step: float, var_M_on: str) -> tuple[int, dict]:
    row = metrics.iloc[int(np.abs(metrics.index.to_numpy() - step).argmin())]
    op = {name: float(row[name]) for name in KEYS}
    if var_M_on == "raw":
        op["var_M_on"] = float(row["var_M_on_raw"])
    return int(row.name), op


def gradients_for_L(L: int, run_ids: list[str], var_M_on: str, threads: int) -> tuple[list, list]:
    torch.set_num_threads(threads)
    rows, negatives, loss = [], [], None
    for run_id in run_ids:
        run = RunData(SOURCE, run_id, base_dir=CODE / "data")
        config, metrics = run.config, run.metrics.dropna(subset=["loss", *KEYS, "var_M_on_raw"])
        loss = make_loss(config, keep_drop()) if loss is None else loss
        T = real_learning_time(run, FRACTION)
        for point, step in point_steps(metrics["loss"], config):
            if math.isnan(step):
                continue
            logged, op = run_order_params(metrics, step, var_M_on)
            meta = {"L": L, "seed": config.extra_args.seed, "run": run_id, "point": point, "step": logged,
                    "t_over_T": logged / T}
            rows += [{**meta, **row} for row in split_rows(loss, op)]
            for cut, groups in CUTS.items():
                for group in groups:
                    if cut == "B" and group in CUTS["A"]:
                        continue
                    negatives += [{**meta, "dropped": group, "moment": moment, "mass": mass}
                                  for moment, mass in negative_moments(loss, op, keep_drop(group)).items()]
    print(f"gradients L={L}: {len(run_ids)} runs", flush=True)
    return rows, negatives


def gradients(args):
    by_L = {L: [run.run_id for run in by_seed.values()] for L, by_seed in source_runs().items()
            if not args.Ls or L in args.Ls}
    rows, negatives = [], []
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(gradients_for_L, L, ids, args.var_M_on, args.threads) for L, ids in by_L.items()]
        for future in as_completed(futures):
            r, n = future.result()
            rows += r
            negatives += n
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["L", "seed", "point", "cut", "part", "param"]).to_csv(
        OUT / "gradient_split.csv", index=False)
    pd.DataFrame(negatives).sort_values(["L", "seed", "point", "dropped", "moment"]).to_csv(
        OUT / "negative_moments.csv", index=False)


# ---- flows ---------------------------------------------------------------------------------

def flow_path(mode: str, L: int) -> Path:
    return OUT / "flows" / f"{mode.replace(':', '_').replace(',', '__')}_L{L}"


def run_flow(mode: str, L: int, config_dict: dict, reference: dict, threads: int) -> dict:
    """One flow of mode at L, written to flow_path; returns its JSON record."""
    torch.set_num_threads(threads)
    config = OmegaConf.create(config_dict)
    V, K = config.model_args.vocab_size, config.data_args.K
    keep, keys = keep_of(mode), [name for name in KEYS if name not in zeroed_of(mode)]
    loss = make_loss(config, keep)
    start = {name: value for name, value in typical_order_params(config, 1).items() if name in keys}
    rates = exact_rates(config, keys)
    target = math.log(V) - STOP_FRACTION * (math.log(V) - asymptotic_loss(V, L, K))
    cap = MAX_FACTOR * reference["length"]
    clock, state, offset, chunk, pieces, evaluations = time.perf_counter(), dict(start), 0, reference["length"], [], [0]

    def counted(order_params):
        evaluations[0] += 1
        return loss(order_params)

    while offset < cap:
        length = min(chunk, cap - offset)
        flow = integrate(counted, state, rates, steps=length,
                         record_steps=np.unique(np.linspace(0, length, POINTS_PER_CHUNK + 1).round()), method="LSODA")
        flow.index = flow.index + offset
        pieces.append(flow if offset == 0 else flow.iloc[1:])
        if flow["loss"].min() <= target or flow.index[-1] < offset + length - 1e-6:
            break
        state = {name: float(flow.iloc[-1][name]) for name in start}
        offset += length
        chunk *= 2 if offset >= CONSTANT_CHUNKS * reference["length"] else 1
    trajectory = pd.concat(pieces)
    T = learning_time(trajectory["loss"], V, L, K, FRACTION)
    record = {"mode": mode, "L": L, "T_star": T, "censored": bool(math.isnan(T)), "steps": float(trajectory.index[-1]),
              "final_progress": float((math.log(V) - trajectory["loss"].iloc[-1]) / (math.log(V) - asymptotic_loss(V, L, K))),
              "T_sgd_median": reference[FRACTION], "sgd_length": reference["length"],
              "loss_evaluations": evaluations[0], "seconds": time.perf_counter() - clock}
    M_on, M_0 = trajectory["M_on"], start["M_on"]
    horizon = trajectory.index[-1] if math.isnan(T) else T
    record |= {"M_on_min_ratio": float(M_on[M_on.index <= horizon].abs().min() / abs(M_0)),   # |M_on| (it may change sign)
               "M_on_half_ratio": float(np.interp(horizon / 2, M_on.index, M_on) / M_0)}
    if mode.split(",zero:")[0] not in ("full", "full_table2", "signal"):
        masses = {}
        for time_ in NEGATIVE_TIMES:
            row = trajectory.iloc[int(np.abs(trajectory.index.to_numpy() - time_ * horizon).argmin())]
            for moment, mass in negative_moments(loss, {name: float(row[name]) for name in keys}, keep).items():
                masses[moment] = max(masses.get(moment, 0.0), mass)
        record["negative_mass"] = masses
    path = flow_path(mode, L)
    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(path.with_suffix(".csv"))
    path.with_suffix(".json").write_text(json.dumps(record, indent=1))
    print(f"L={L:4d} {mode:28s} T*={T:9.1f} (SGD {reference[FRACTION]:8.1f})  {record['seconds']:6.0f} s", flush=True)
    return record


def references(Ls) -> dict:
    """{L: (config dict, reference)}: the config of the first seed and the SGD medians."""
    out = {}
    for L, by_seed in source_runs().items():
        if Ls and L not in Ls:
            continue
        config = by_seed[min(by_seed)].config
        lengths = [int(run.metrics["loss"].dropna().index[-1]) for run in by_seed.values()]
        reference = {"length": int(statistics.median(lengths)),
                     FRACTION: statistics.median(real_learning_time(run, FRACTION) for run in by_seed.values())}
        out[L] = (OmegaConf.to_container(config, resolve=True), reference)
    return out


def flows(args):
    for mode in args.modes:
        keep_of(mode), zeroed_of(mode)               # fails early on an unknown group or variance
    refs = references(args.Ls)
    jobs = [(mode, L) for mode in args.modes for L in sorted(refs) if not flow_path(mode, L).with_suffix(".json").exists()]
    print(f"{len(jobs)} flows, {args.workers} workers x {args.threads} threads", flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(run_flow, mode, L, *refs[L], args.threads): (mode, L) for mode, L in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:               # keep the others going
                print(f"FAILED {futures[future]}: {error!r}", flush=True)


def flow_gradients(args):
    refs = references(args.Ls)
    rows = []
    for L in sorted(refs):
        path = flow_path("full", L).with_suffix(".csv")
        if not path.exists():
            continue
        config = OmegaConf.create(refs[L][0])
        trajectory, loss = pd.read_csv(path, index_col=0), make_loss(config, keep_drop())
        T = progress_step(trajectory["loss"], config, FRACTION)
        for point, step in point_steps(trajectory["loss"], config):
            if math.isnan(step):
                continue
            row = trajectory.iloc[int(np.abs(trajectory.index.to_numpy() - step).argmin())]
            meta = {"L": L, "point": point, "step": float(row.name), "t_over_T": float(row.name) / T}
            rows += [{**meta, **r} for r in split_rows(loss, {name: float(row[name]) for name in KEYS})]
        print(f"flow gradients L={L}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "flow_gradient_split.csv", index=False)


# ---- summary ----------------------------------------------------------------------------

def log_slope(L, values) -> float:
    L, values = np.asarray(L, dtype=float), np.abs(np.asarray(values, dtype=float))
    good = values > 0
    return float(np.polyfit(np.log(L[good]), np.log(values[good]), 1)[0]) if good.sum() >= 2 else math.nan


def gradient_summary():
    """Per (L, point, param, part), over seeds: the median and quartiles of |part| / |signal|
    (the means) or / |total| (the variances); the median of part / signal (sign-free: the signs
    of the means differ between seeds, and part and signal flip together); the fraction of seeds
    where the part opposes the signal (part x signal < 0) and where it shrinks |theta|
    (part x theta > 0, the friction); then the log-slope in L of the median |share| per segment."""
    data = pd.read_csv(OUT / "gradient_split.csv")
    wide = data.pivot_table(index=["L", "seed", "point", "param", "theta"], columns="part", values="value").reset_index()
    parts = [part for part in wide.columns if part not in ("L", "seed", "point", "param", "theta", "signal", "total")]
    rows = []
    for (L, point, param), group in wide.groupby(["L", "point", "param"]):
        scale = (group["signal"] if param in ORDER_PARAMS else group["total"]).abs()
        for part in parts:
            share = (group[part] / scale).abs()
            rows.append({"L": L, "point": point, "param": param, "part": part,
                         "normalised_by": "signal" if param in ORDER_PARAMS else "total",
                         "median": share.median(), "q25": share.quantile(0.25), "q75": share.quantile(0.75),
                         "median_vs_signal": (group[part] / group["signal"]).median() if param in ORDER_PARAMS
                         else math.nan,
                         "opposes_signal": float((group[part] * group["signal"] < 0).mean())
                         if param in ORDER_PARAMS else math.nan,
                         "shrinks": float((group[part] * group["theta"] > 0).mean())})
    summary = pd.DataFrame(rows)
    slopes = []
    for (point, param, part), group in summary.groupby(["point", "param", "part"]):
        for name, mask in (("slope_L<=192", group["L"] <= 192), ("slope_L>=192", group["L"] >= 192)):
            slopes.append({"point": point, "param": param, "part": part, "segment": name,
                           "slope": log_slope(group["L"][mask], group["median"][mask])})
    summary.to_csv(OUT / "gradient_summary.csv", index=False)
    pd.DataFrame(slopes).pivot_table(index=["point", "param", "part"], columns="segment", values="slope").to_csv(
        OUT / "gradient_slopes.csv")


def T_star_summary():
    records = [json.loads(path.read_text()) for path in sorted((OUT / "flows").glob("*.json"))]
    table = pd.DataFrame([{key: value for key, value in record.items() if key != "negative_mass"}
                          | {f"negative_{k}": v for k, v in record.get("negative_mass", {}).items()} for record in records])
    full = table[table["mode"] == "full"].set_index("L")["T_star"]
    table["ratio_to_full"] = table["T_star"] / table["L"].map(full)
    table["ratio_to_sgd"] = table["T_star"] / table["T_sgd_median"]
    table.sort_values(["mode", "L"]).to_csv(OUT / "T_star.csv", index=False)


def summary(args):
    if (OUT / "gradient_split.csv").exists():
        gradient_summary()
    if (OUT / "flows").exists():
        T_star_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["gradients", "flows", "flow-gradients", "summary"])
    parser.add_argument("--Ls", type=int, nargs="*", default=[])
    parser.add_argument("--modes", nargs="+", default=["full"],
                        help='"full", "signal", "drop:G[+G...]", "add:G[+G...]"; groups: ' + ", ".join(GROUPS))
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", 8)))
    parser.add_argument("--var-M-on", dest="var_M_on", default="raw", choices=["raw", "smooth"])
    args = parser.parse_args()
    args.Ls = set(args.Ls)
    {"gradients": gradients, "flows": flows, "flow-gradients": flow_gradients, "summary": summary}[args.command](args)
