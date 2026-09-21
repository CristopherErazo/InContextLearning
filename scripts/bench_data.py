"""Time the batch generator: loop version vs vectorised, CPU vs CUDA.

    uv run python -u scripts/bench_data.py                 # the default grid
    uv run python -u scripts/bench_data.py --step          # + a full train step
    uv run python -u scripts/bench_data.py --config V=512 L=256

The point of this script is to decide `data_args.gen_device`. The old
loop-based generator dispatched ~8 kernels per sequence position, which is why
CPU beat CUDA for it; the vectorised one dispatches about a dozen in total, so
the answer may well flip. Run it on the machine that actually trains -- the
answer is a property of that machine, not of the code.

`generate_icl_batch_loop` is imported from `tests/reference_data.py`, the only
place the pre-refactor implementation still lives.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from icl import MinimalTransformer, compute_loss, generate_icl_batch, load_config  # noqa: E402
from reference_data import generate_icl_batch_loop  # noqa: E402

# (B, V, L, K): the configurations the sweeps actually run, plus a large one.
GRID = [
    (145, 128, 128, 25),
    (481, 512, 128, 102),
    (256, 128, 256, 25),
    (1000, 512, 512, 102),
]


def timeit(fn, n: int, device: str) -> float:
    """Mean seconds per call, with CUDA synchronised around the timing."""
    for _ in range(2):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def batch_mb(batch: dict) -> float:
    return sum(v.numel() * v.element_size() for v in batch.values()) / 1e6


def bench_generators(grid, devices, n: int) -> None:
    head = f"{'B':>6} {'V':>5} {'L':>5} {'K':>5} {'device':>7} | {'loop':>9} {'vector':>9} {'vec (no stats)':>15} | {'speedup':>8} | {'MB loop':>8} {'MB vec':>8}"
    print(head)
    print("-" * len(head))
    for B, V, L, K in grid:
        for device in devices:
            t_loop = timeit(lambda: generate_icl_batch_loop(B, V, L, K, device=device), n, device)
            t_vec = timeit(lambda: generate_icl_batch(B, V, L, K, device=device), n, device)
            t_seq = timeit(lambda: generate_icl_batch(B, V, L, K, stats=False, device=device), n, device)
            mb_loop = batch_mb(generate_icl_batch_loop(B, V, L, K, device=device))
            mb_vec = batch_mb(generate_icl_batch(B, V, L, K, stats=False, device=device))
            print(f"{B:6d} {V:5d} {L:5d} {K:5d} {device:>7} | "
                  f"{t_loop * 1e3:8.2f}m {t_vec * 1e3:8.2f}m {t_seq * 1e3:14.2f}m | "
                  f"{t_loop / t_seq:7.1f}x | {mb_loop:8.1f} {mb_vec:8.1f}")


def bench_train_step(overrides: list[str], n: int) -> None:
    """End-to-end: how much of a training step is data generation, per device."""
    cfg = load_config([f"model_args.{o}" if o.split("=")[0] in
                       ("vocab_size", "seq_len", "d_model") else o for o in overrides])
    V, L = cfg.model_args.vocab_size, cfg.model_args.seq_len
    B, K = cfg.data_args.batch_size, cfg.data_args.K
    train_device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MinimalTransformer(cfg.model_args).to(train_device)
    model.initialize_model()
    loss_fn = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.optim_args.lr)

    print(f"\ntrain step: B={B} V={V} L={L} K={K} d={cfg.model_args.d_model} on {train_device}")
    for gen_device in dict.fromkeys(["cpu", train_device]):
        def step():
            batch = generate_icl_batch(B, V, L, K, stats=False, device=gen_device)
            loss = compute_loss(model, batch, loss_fn, train_device)
            opt.zero_grad()
            loss.backward()
            opt.step()

        t_gen = timeit(lambda: generate_icl_batch(B, V, L, K, stats=False, device=gen_device), n, gen_device)
        t_all = timeit(step, n, train_device)
        print(f"  gen_device={gen_device:>4}: {t_all * 1e3:8.2f} ms/step "
              f"(generation {t_gen * 1e3:6.2f} ms = {t_gen / t_all:4.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10, help="timed repetitions per cell")
    ap.add_argument("--step", action="store_true", help="also time a full training step")
    ap.add_argument("--config", nargs="*", default=[], help="overrides for --step, e.g. vocab_size=512")
    args = ap.parse_args()

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print(f"torch {torch.__version__} | devices: {', '.join(devices)}")
    if "cuda" in devices:
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    bench_generators(GRID, devices, args.n)
    if args.step:
        bench_train_step(args.config, args.n)
