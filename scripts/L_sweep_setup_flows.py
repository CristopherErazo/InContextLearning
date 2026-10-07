"""Effective-model flows for the setups of notebooks/26-10-05_Ansatz-check.ipynb, over the L-sweep of
L_sweep_B1024 (d = inf, batch 1024, L = 32 ... 1024), each flow saved as a TrackLab run that
RunData reads like a training run:

    run = RunData("setup_flows_B1024", "run_007", base_dir="data")
    run.config.flow_args          # setup, keys, init, sign, source_run, method, num_mus, noise_samples
    run.metrics                   # the order parameters and "loss" (the effective loss) per step
    run.profile(step)             # "M_profile" (profile setups), every PROFILE_EVERY recorded steps
    learning_time(run.metrics["loss"], V, L, K, 0.2)
    run.reader.load_results(run.run_id)   # T_f20, T_f50 (f = 0.2, 0.5) of the flow and of the reference, cost

Initial conditions (--init):
- "typical": seed-free. Every mean at the typical size of a block mean at initialisation,
  s_X / sqrt(n) (n its support size; s_M = s_G = sigma_0, s_Q = sigma_0^2), all positive except
  G_on, which takes the sign --signs of M_on Q_on G_on; every block variance at s_X^2; the profile
  flat at M_on (the spread of the sub-diagonal is in var_M_on). The reference T* is the median over
  the seeds of L_sweep_B1024 at that L.
- "measured": the order parameters measured on the step-0 matrices of the L_sweep_B1024 run with
  that seed (--seeds), the raw sub-diagonal as "M_profile" (it already carries the spread of those
  entries, so a setup with "var_M_on" too counts it twice). The reference T* is that run's.

For each setup exactly its keys move, with `exact_rates` (the d = inf SGD rates: a mean
eta_0 kappa_X / n, the profile eta_0 per entry, a variance 4 eta_0 kappa_X / n times itself); every
other order parameter is absent, i.e. 0. The flow is integrated in chunks (the first 2x the
reference run length, then doubling) until the loss has gone STOP_FRACTION of the way from log V to
asymptotic_loss, the stop of the L_sweep_B1024 runs, or MAX_FACTOR x the reference length.

Solver: the noise setups are stiff (the variance rates are large, e.g. 560 per step for var_Q_on
whatever L, so the variances collapse within a few hundred steps and an explicit solver then crawls:
RK45 needed ~0.6 loss evaluations per step at L = 32, hours for one flow). LSODA estimates its
Jacobian by finite differences, one evaluation per state entry: cheap without a profile, 2-3x slower
than RK45 with the 511-1023 profile entries at L >= 512. Hence "auto": LSODA up to MAX_LSODA_SIZE
moving entries, RK45 above. num_mus = 64 and noise_samples = 16 change T* by <= 0.6% against
noise_samples = 32 (seed 13, L = 32, 128, 1024).

    nohup setsid uv run --no-sync python -u scripts/L_sweep_setup_flows.py > logs/setup_flows.log 2>&1 &
    uv run --no-sync python -u scripts/L_sweep_setup_flows.py --init measured --seeds 13 --Ls 64 512

A flow already completed in the experiment (same init, sign or seed, L, setup) is skipped, so the
script can be restarted; an interrupted flow leaves a run without results, which is redone.
"""
import argparse
import math
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tracklab import ExperimentReader, ExperimentTracker

from icl import (ORDER_PARAMS, PROFILE, EffectiveLoss, RunData, asymptotic_loss, exact_rates, integrate,
                 learning_time, support_sizes)

CODE = Path(__file__).resolve().parents[1]
DATA = CODE / "data"
SOURCE = "L_sweep_B1024"

SETUPS = {"signal_only": ["M_on", "Q_on", "G_on"]}
SETUPS["means_only"] = SETUPS["signal_only"] + ["M_off", "Q_T", "G_T"]
SETUPS["means_&_noise"] = SETUPS["means_only"] + ["var_M_on", "var_M_off", "var_Q_on", "var_Q_T", "var_Q_N",
                                                  "var_G_on", "var_G_T", "var_G_N"]
SETUPS["means_&_profile"] = SETUPS["means_only"] + ["M_profile"]
SETUPS["means,_noise_&_profile"] = SETUPS["means_&_noise"] + ["M_profile"]

FRACTIONS = (0.2, 0.5)
STOP_FRACTION, MAX_FACTOR, POINTS_PER_CHUNK, PROFILE_EVERY = 0.5, 50, 200, 5
MAX_LSODA_SIZE = 300                     # moving entries: profile at L <= 256 + 14 scalars


def source_runs() -> dict:
    """{L: {seed: RunData}} of the d = inf runs of SOURCE."""
    runs = {}
    for run_id in ExperimentReader(SOURCE, base_dir=DATA).list_runs():
        run = RunData(SOURCE, run_id, base_dir=DATA)
        if run.config.model_args.infinite_d:
            runs.setdefault(run.config.model_args.seq_len, {})[run.config.extra_args.seed] = run
    return runs


def real_learning_time(run: RunData, fraction: float) -> float:
    model_args, K = run.config.model_args, run.config.data_args.K
    return learning_time(run.metrics["loss"].dropna(), model_args.vocab_size, model_args.seq_len, K, fraction)


def typical_order_params(config, sign: int) -> dict:
    """The typical sizes at initialisation: means s_X / sqrt(n) (G_on with the sign of M_on Q_on G_on),
    block variances s_X^2, the profile flat at M_on."""
    L, V, K, sigma_0 = config.model_args.seq_len, config.model_args.vocab_size, config.data_args.K, config.model_args.sigma_0
    scale = {"M": sigma_0, "Q": sigma_0 ** 2, "G": sigma_0}
    sizes = support_sizes(L, V, K)
    values = {name: scale[matrix] / math.sqrt(sizes[name]) for name, (matrix, _) in ORDER_PARAMS.items()}
    values["G_on"] *= sign
    values |= {f"var_{matrix}_{block}": scale[matrix] ** 2
               for matrix, blocks in (("M", ("on", "off")), ("Q", ("on", "T", "N")), ("G", ("on", "T", "N")))
               for block in blocks}
    values[PROFILE] = torch.full((L - 1,), values["M_on"], dtype=torch.float64)
    return values


def solver_for(start: dict, solver: str) -> str:
    if solver != "auto":
        return solver
    size = sum(torch.as_tensor(value).numel() for value in start.values())
    return "LSODA" if size <= MAX_LSODA_SIZE else "RK45"


def completed(experiment: str) -> set:
    """(init, sign or seed, L, setup) of the flows of `experiment` that finished (have results)."""
    reader, done = ExperimentReader(experiment, base_dir=DATA), set()
    for run_id in reader.list_runs():
        if "T_f20" in reader.load_results(run_id):
            config = RunData(experiment, run_id, base_dir=DATA).config
            flow = config.flow_args
            label = flow.sign if flow.init == "typical" else config.extra_args.seed
            done.add((flow.init, label, config.model_args.seq_len, flow.setup))
    return done


def run_flow(tracker, config, loss, start: dict, setup: str, solver: str, reference: dict) -> None:
    """Integrate one flow in chunks, tracking the order parameters and the loss of every recorded step,
    until the loss is STOP_FRACTION of the way to asymptotic_loss or MAX_FACTOR x the reference length."""
    L, V, K = config.model_args.seq_len, config.model_args.vocab_size, config.data_args.K
    keys = SETUPS[setup]
    rates = exact_rates(config, keys)
    method = solver_for(start, solver)
    target_loss = math.log(V) - STOP_FRACTION * (math.log(V) - asymptotic_loss(V, L, K))
    calls = [0]

    def counted(order_params):
        calls[0] += 1
        return loss(order_params)

    clock = time.perf_counter()
    with tracker.start_run(config, artifacts=PROFILE in keys) as run:
        log = run.get_logger(log_to_terminal=False, log_to_file=True)
        state, offset, chunk, recorded, losses = dict(start), 0, 2 * reference["length"], 0, []
        while offset < MAX_FACTOR * reference["length"]:
            length = min(chunk, MAX_FACTOR * reference["length"] - offset)
            record_steps = np.unique(np.linspace(0, length, POINTS_PER_CHUNK + 1).round())
            chunk_clock = time.perf_counter()
            flow = integrate(counted, state, rates, steps=length, record_steps=record_steps, method=method)
            flow.index = flow.index + offset
            new = flow if offset == 0 else flow.iloc[1:]
            losses.append(new["loss"])
            for step, row in new.iterrows():
                run.track_metric(int(step), **{name: float(value) for name, value in row.items() if name != PROFILE})
                if PROFILE in row and (recorded % PROFILE_EVERY == 0 or step == flow.index[-1]):
                    run.track_artifact(np.asarray(row[PROFILE]), step=int(step), group="profile", name="M_profile")
                recorded += 1
            message = (f"L={L} {setup}: steps {offset}-{int(flow.index[-1])} ({method}), loss {flow['loss'].iloc[-1]:.5f}, "
                       f"{time.perf_counter() - chunk_clock:.0f} s")
            log.info(message)
            print("    " + message, flush=True)
            if flow["loss"].min() <= target_loss or flow.index[-1] < offset + length - 1e-6:   # learned, or solver stopped
                break
            last = flow.iloc[-1]
            state = {name: torch.as_tensor(last[name]) if name == PROFILE else float(last[name]) for name in start}
            offset += length
            chunk *= 2
        loss_curve = pd.concat(losses)
        T = {f: learning_time(loss_curve, V, L, K, f) for f in FRACTIONS}
        run.track_results(**{f"T_f{round(100 * f)}": T[f] for f in FRACTIONS},
                          **{f"T_reference_f{round(100 * f)}": reference[f] for f in FRACTIONS},
                          reference=reference["kind"], reference_seeds=reference["seeds"], solver=method,
                          steps=int(loss_curve.index[-1]), final_loss=float(loss_curve.iloc[-1]), loss_evaluations=calls[0],
                          seconds=time.perf_counter() - clock)
    print(f"L={L:4d} {setup:24s} T*={T[0.2]:9.1f} (reference {reference[0.2]:8.1f})  {calls[0]:5d} evals, "
          f"{time.perf_counter() - clock:6.0f} s", flush=True)


def main(args):
    experiment = args.experiment
    tracker = ExperimentTracker(experiment, base_dir=DATA)
    done = completed(experiment) if (DATA / experiment).exists() else set()
    sweep = source_runs()
    for L in sorted(sweep):
        if args.Ls and L not in args.Ls:
            continue
        by_seed = sweep[L]
        labels = args.signs if args.init == "typical" else [seed for seed in args.seeds if seed in by_seed]
        for label in labels:
            source = by_seed[min(by_seed)] if args.init == "typical" else by_seed[label]
            config = OmegaConf.merge(source.config, {})            # a copy: V, L, K, beta, sigma_0, alpha_lr
            V, K, beta = config.model_args.vocab_size, config.data_args.K, config.model_args.beta
            if args.init == "typical":
                start = typical_order_params(config, label)
                lengths = [int(run.metrics["loss"].dropna().index[-1]) for run in by_seed.values()]
                reference = {"kind": "median over seeds", "seeds": len(by_seed),
                             "length": int(statistics.median(lengths)),
                             **{f: statistics.median(real_learning_time(run, f) for run in by_seed.values())
                                for f in FRACTIONS}}
            else:
                start = source.order_params(0, profile=True, variances=True)
                reference = {"kind": "run", "seeds": 1, "length": int(source.metrics["loss"].dropna().index[-1]),
                             **{f: real_learning_time(source, f) for f in FRACTIONS}}
            mus = np.unique(np.linspace(1, L, args.num_mus).round().astype(int))
            loss = EffectiveLoss(V, L, K, beta, method=args.method, mus=mus, device=args.device,
                                 noise_samples=args.noise_samples)
            for setup in args.setups:
                if (args.init, label, L, setup) in done:
                    continue
                config.extra_args.update(experiment_name=experiment, base_dir=str(DATA), log_to_terminal=False,
                                         total_steps=MAX_FACTOR * reference["length"],
                                         seed=None if args.init == "typical" else label)
                config.flow_args = OmegaConf.merge(config.flow_args, {
                    "setup": setup, "keys": SETUPS[setup], "init": args.init,
                    "sign": label if args.init == "typical" else 1,
                    "source_run": f"{SOURCE}/{source.run_id}", "method": args.method, "num_mus": len(mus),
                    "noise_samples": args.noise_samples})
                run_flow(tracker, config, loss, {name: start[name] for name in SETUPS[setup]}, setup, args.solver,
                         reference)


if __name__ == "__main__":
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="typical", choices=["typical", "measured"])
    parser.add_argument("--signs", type=int, nargs="+", default=[1], choices=[1, -1],
                        help="typical init: signs of M_on Q_on G_on")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13], help="measured init: seeds of the source runs")
    parser.add_argument("--Ls", type=int, nargs="*", default=[], help="default: every L of the sweep")
    parser.add_argument("--setups", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--experiment", default="setup_flows_B1024")
    parser.add_argument("--method", default="mean", choices=["mean", "mc"])
    parser.add_argument("--num-mus", type=int, default=64)
    parser.add_argument("--noise-samples", type=int, default=16)
    parser.add_argument("--solver", default="auto", choices=["auto", "LSODA", "RK45", "BDF", "Radau"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.Ls = set(args.Ls)
    main(args)
