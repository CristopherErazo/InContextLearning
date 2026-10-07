"""Many-seed d = inf L-sweep: for each seed in turn, every L of the grid, so the seeds
finished so far form a complete sweep at any time. The settings are those of the
d = inf runs of L_sweep_reduced, with a larger batch.

    nohup setsid uv run --no-sync python -u scripts/L_sweep_seeds.py > logs/L_sweep_B1024.log 2>&1 &

Each run stops when the loss has gone half of the way from log V to asymptotic_loss
(so learning_time at any fraction <= 0.5 exists), at most MAX_STEPS steps; accuracy is
still logged. A (seed, L) already completed in the experiment is skipped, so the
script can be restarted after an interruption.
"""
import json
import math
import sys
from pathlib import Path

from omegaconf import OmegaConf

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "scripts"))
from train import train                                      # noqa: E402
from icl import TrainerArgs, asymptotic_loss, set_seed       # noqa: E402

EXPERIMENT = "L_sweep_B1024"
TEMPLATE = CODE / "data/L_sweep_reduced/run_028/config.json"    # d = inf, init=sample
LS = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
SEEDS = range(1, 31)
BATCH_SIZE, STOP_FRACTION, MAX_STEPS = 1024, 0.5, 25000


def completed() -> set:
    done = set()
    for run_dir in (CODE / "data" / EXPERIMENT).glob("run_*"):
        if (run_dir / "results.json").exists():
            config = json.loads((run_dir / "config.json").read_text())
            done.add((config["extra_args"]["seed"], config["model_args"]["seq_len"]))
    return done


template = json.loads(TEMPLATE.read_text())
done = completed()
for seed in SEEDS:
    for L in LS:
        if (seed, L) in done:
            continue
        cfg = OmegaConf.merge(OmegaConf.structured(TrainerArgs()), template)
        V, K = cfg.model_args.vocab_size, cfg.data_args.K
        cfg.model_args.seq_len = L
        cfg.data_args.batch_size = BATCH_SIZE
        cfg.extra_args.update(seed=seed, alpha_steps=None, total_steps=MAX_STEPS, stop_at_accuracy=None,
                              stop_at_loss=math.log(V) - STOP_FRACTION * (math.log(V) - asymptotic_loss(V, L, K)),
                              experiment_name=EXPERIMENT, base_dir=str(CODE / "data"))
        cfg.extra_args.seed, _ = set_seed(seed)
        print(f"seed={seed} L={L} stop_at_loss={cfg.extra_args.stop_at_loss:.4f}", flush=True)
        try:
            train(cfg, log_to_terminal=False)
        except Exception as error:                            # keep the sweep going
            print(f"  FAILED seed={seed} L={L}: {error!r}", flush=True)
