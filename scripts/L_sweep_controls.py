"""Reruns of the d = inf runs of L_sweep_reduced for the effective L-sweep
(scripts/L_sweep_flows.py), each with its own seed (same draw), stopped when the
loss has gone half of the way from log V to asymptotic_loss. One experiment per
kind of run:

- real:  the run as it was, but with the loss stop (<prefix>_loss_stop). At the
         original sigma_0 only the runs that stopped on the accuracy before the
         loss moved (seed 3, L = 64, 128) are rerun;
- none:  the draw projected on the extended ansatz, i.e. the same order
         parameters without the spread (<prefix>_ansatz_init);
- QG, M, ...: the same, keeping those matrices as drawn (<prefix>_spread_<X>).

<prefix> is L_sweep_reduced, or L_sweep_reduced_sigma<s> with --sigma0 s
(same seed, so the same draw scaled: m, g by s / sigma_0, q by its square).

    uv run python -u scripts/L_sweep_controls.py                    # real + none
    uv run python -u scripts/L_sweep_controls.py QG M
    uv run python -u scripts/L_sweep_controls.py --sigma0 0.5       # real + none at sigma_0 = 0.5
    uv run python -u scripts/L_sweep_controls.py none --runs run_011 run_026   # only these source runs
"""
import argparse
import json
import math
import sys
from pathlib import Path

from omegaconf import OmegaConf

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "scripts"))
from train import train                                      # noqa: E402
from icl import TrainerArgs, asymptotic_loss, set_seed       # noqa: E402

STOP_FRACTION, MAX_STEPS = 0.5, 20000
RERUN = {"run_041", "run_042"}

parser = argparse.ArgumentParser()
parser.add_argument("kinds", nargs="*", default=["real", "none"])
parser.add_argument("--sigma0", type=float, default=None)
parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
parser.add_argument("--runs", nargs="*", default=None, help="source runs of L_sweep_reduced (default: all d = inf)")
args = parser.parse_args()
prefix = "L_sweep_reduced" if args.sigma0 is None else f"L_sweep_reduced_sigma{args.sigma0:g}"

sources = []
for path in sorted((CODE / "data/L_sweep_reduced").glob("run_*/config.json")):
    config = json.loads(path.read_text())
    if config["model_args"]["infinite_d"] and (args.runs is None or path.parent.name in args.runs):
        sources.append((path.parent.name, config))

for kind in args.kinds:
    for name, config in sources:
        if kind == "real" and args.sigma0 is None and name not in RERUN:
            continue
        cfg = OmegaConf.merge(OmegaConf.structured(TrainerArgs()), config)
        V, L, K = cfg.model_args.vocab_size, cfg.model_args.seq_len, cfg.data_args.K
        cfg.extra_args.stop_at_accuracy = None
        cfg.extra_args.stop_at_loss = math.log(V) - STOP_FRACTION * (math.log(V) - asymptotic_loss(V, L, K))
        if args.sigma0 is not None:
            cfg.model_args.sigma_0 = args.sigma0
            cfg.extra_args.total_steps = args.max_steps
        if kind == "real":
            experiment = f"{prefix}_loss_stop"
        else:
            experiment = f"{prefix}_ansatz_init" if kind == "none" else f"{prefix}_spread_{kind}"
            cfg.model_args.ansatz_init, cfg.model_args.ansatz_keep_spread = True, "" if kind == "none" else kind
            cfg.extra_args.total_steps = args.max_steps
        cfg.extra_args.experiment_name = experiment
        cfg.extra_args.base_dir = str(CODE / "data")
        cfg.extra_args.seed, message = set_seed(cfg.extra_args.seed)
        print(f"{experiment} <- {name}: L={L} seed={cfg.extra_args.seed} sigma_0={cfg.model_args.sigma_0} "
              f"stop_at_loss={cfg.extra_args.stop_at_loss:.4f}", flush=True)
        train(cfg, log_to_terminal=False)
