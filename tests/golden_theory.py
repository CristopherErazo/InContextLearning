"""Golden values of the public `icl.theory` usages, for tests/test_golden.py.

`compute(inputs)` evaluates every public function on fixed inputs (CPU, float64, fixed
seeds) and returns a flat {name: tensor}. The inputs are two parameter points at the size
of full_ansatz/run_001 (V=128, L=512, K=25, beta=0.25):

- "run001": its last-step order parameters (six means, the measured profile, the eight
  block and three pooled variances), stored in the fixture since data/ is not versioned;
- "mid": bulk and previous-token attention comparable, a decreasing profile and all eight
  variances non-zero.

Write the fixture (once, or after a documented change) with

    cd code && .venv/bin/python tests/golden_theory.py

which reads run_001 from data/full_ansatz and saves tests/golden/theory_golden.pt.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from icl import (LEVELS, ORDER_PARAMS, PROFILE, EffectiveLoss, TrainerArgs, ansatz_logits, ansatz_matrices,
                 asymptotic_loss, at_level, block_variance, compute_derived_args, exact_rates, generate_icl_batch,
                 integrate, learning_time, mean_variables, measure_order_params, measure_variables,
                 measure_variances, noise_variances, non_trigger_loss, sample_logits, sample_variables, support_sizes,
                 variance_sizes)
from icl.theory.ansatz import VARIANCES

FIXTURE = Path(__file__).parent / "golden" / "theory_golden.pt"
CHANGES = Path(__file__).parent / "golden" / "theory_golden_changes.pt"     # the agreed new values of moved ones
V, L, K, BETA = 128, 512, 25, 0.25
BATCH_MUS = (256, 512)
LOSS_MUS = tuple(range(32, L + 1, 32))
GRID = ((64, 0), (64, 1), (192, 0), (192, 1), (192, 3), (384, 2), (512, 0), (512, 4))     # (mu, ell)

MID_MEANS = {"M_on": 1.0, "M_off": 0.01, "Q_on": 3.0, "Q_T": 0.5, "G_on": 2.0, "G_T": -0.7}
MID_SDS = {"M_on": 0.3, "M_off": 0.05, "Q_on": 0.8, "Q_T": 0.6, "Q_N": 0.4, "G_on": 0.5, "G_T": 0.6, "G_N": 0.3}


def mid_point() -> dict:
    profile = MID_MEANS["M_on"] * (1.6 - 1.2 * torch.arange(2, L + 1, dtype=torch.float64) / L)
    return {**MID_MEANS, PROFILE: profile, **{"var_" + name: sd ** 2 for name, sd in MID_SDS.items()}}


def run_config():
    """The TrainerArgs of run_001 that `exact_rates` reads (plain SGD, alpha_lr = 3000, sigma_0 = 1)."""
    config = OmegaConf.structured(TrainerArgs())
    config.model_args.update(vocab_size=V, seq_len=L, beta=BETA, backend="reduced", sigma_0=1.0)
    config.data_args.rho = 0.2
    config.optim_args.alpha_lr = 3000.0
    return compute_derived_args(config)


def inputs_from_run(base_dir: str = "data") -> dict:
    from icl import RunData
    run = RunData("full_ansatz", "run_001", base_dir=base_dir)
    op = run.order_params("last", profile=True, variances=True)
    op.pop("var_M_on_raw")
    return {"run001": {name: (value.clone() if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float64))
                       for name, value in op.items()},
            "mid": {name: (value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float64))
                    for name, value in mid_point().items()}}


def _table(prefix: str, table) -> dict:
    """The tensor columns; the key columns (code positions < L) as int16 to keep the fixture small."""
    return {f"{prefix}/{name}": (value.to(torch.int16) if name.endswith("_keys") else value).detach().cpu().clone()
            for name, value in table.items() if torch.is_tensor(value)}


def _tensors(prefix: str, values: dict) -> dict:
    return {f"{prefix}/{name}": torch.as_tensor(value).detach().cpu().clone() for name, value in values.items()}


def compute(inputs: dict) -> dict[str, torch.Tensor]:
    """Every golden value, {name: tensor}. Deterministic on the CPU."""
    torch.manual_seed(0)
    out = {}
    out["asymptotic_loss"] = torch.tensor(asymptotic_loss(V, L, K))
    out |= _tensors("support_sizes", support_sizes(L, V, K)) | _tensors("variance_sizes", variance_sizes(L, V, K))
    out["learning_time"] = torch.tensor(learning_time(pd.Series(math.log(V) - 0.002 * torch.arange(100).numpy(),
                                                                index=10 * torch.arange(100).numpy()), V, L, K))

    batch = generate_icl_batch(128, V, L, K, generator=torch.Generator().manual_seed(0))
    measured = measure_variables(batch, BATCH_MUS)
    out |= _table("measure_variables", measured)
    mus, ells = (torch.tensor(column) for column in zip(*GRID))
    means = mean_variables(mus, ells, V, K)
    out |= _table("mean_variables", means)
    sampled = {law: sample_variables(mus.repeat_interleave(8), ells.repeat_interleave(8), V, K, counts=law,
                                     generator=torch.Generator().manual_seed(1)) for law in ("poisson", "multinomial")}
    for law, table in sampled.items():
        out |= _table(f"sample_variables/{law}", table)

    for point, op in inputs.items():
        scalar = {name: value for name, value in op.items() if name != PROFILE}
        out |= _tensors(f"{point}/at_level", {level: torch.tensor([name in at_level(op, level) for name in
                                                                   (*ORDER_PARAMS, PROFILE, *VARIANCES, "var_M")])
                                              for level in LEVELS})
        out |= _tensors(f"{point}/block_variance", {name: float(block_variance(op, name)) for name in VARIANCES})
        out |= _tensors(f"{point}/measure_order_params", measure_order_params(ansatz_matrices(op, L, V, K), K))
        noisy = ansatz_matrices(op, L, V, K, noise=torch.Generator().manual_seed(2))
        out |= _tensors(f"{point}/measure_variances", measure_variances(noisy, K))
        out[f"{point}/ansatz_matrices_noise/M_diag"] = noisy["M"].diagonal(-1)[:64].clone()
        out[f"{point}/ansatz_matrices_noise/Q_row0"] = noisy["Q"][0].clone()

        for source, table in (("measured", measured), ("mean", means), *sampled.items()):
            out[f"{point}/ansatz_logits/{source}"] = ansatz_logits(table, op, BETA, L)
            out[f"{point}/ansatz_logits_scalar/{source}"] = ansatz_logits(table, scalar, BETA, L)
            if source != "mean":
                out |= _tensors(f"{point}/noise_variances/{source}", noise_variances(table, op, BETA, L))
                out |= _tensors(f"{point}/noise_variances_scalar/{source}", noise_variances(table, scalar, BETA, L))
        out[f"{point}/sample_logits"] = sample_logits(measured, op, BETA, L, generator=torch.Generator().manual_seed(3))
        out[f"{point}/non_trigger_loss"] = non_trigger_loss(torch.arange(1, L + 1), op, BETA, L, V, K)

        for method in ("mean", "mc"):
            loss = EffectiveLoss(V, L, K, BETA, method=method, mus=LOSS_MUS, num_samples=64, noise_samples=16)
            for variant, params in (("full", op), ("scalar", scalar), ("S3", at_level(op, "S3")),
                                    ("S5", at_level(op, "S5"))):
                key = f"{point}/EffectiveLoss/{method}/{variant}"
                parts = loss.breakdown(params)
                out |= _tensors(key, {name: parts[name] for name in ("loss", "loss_trigg", "loss_non_trigg")})
                out[f"{key}/cluster_cross_entropy"] = parts["clusters"]["cross_entropy"]
                value, gradient = loss.value_and_grad(params)
                out[f"{key}/value"] = torch.tensor(value)
                out |= _tensors(f"{key}/grad", gradient)

    rates = exact_rates(run_config(), [*ORDER_PARAMS, PROFILE, *VARIANCES])
    out |= _tensors("exact_rates", rates)
    loss = EffectiveLoss(V, L, K, BETA, method="mean", mus=LOSS_MUS)
    start = {name: float(value) for name, value in inputs["run001"].items() if name in ORDER_PARAMS}
    flow = integrate(loss, {name: 0.5 * value for name, value in start.items()},
                     {name: rates[name] for name in ORDER_PARAMS}, steps=200, record_steps=[0, 50, 100, 200])
    out |= _tensors("integrate", {name: torch.tensor(flow[name].to_numpy()) for name in flow.columns})
    return out


if __name__ == "__main__":
    import sys
    if "--changes" in sys.argv:
        # after an agreed change: store the new values of the moved ones (see test_golden.EXPECTED_CHANGES)
        from test_golden import RTOL, _relative_change
        golden = torch.load(FIXTURE)
        values = compute(golden["inputs"])
        changed = {name: values[name] for name, old in golden["values"].items()
                   if _relative_change(values[name], old) > RTOL}
        torch.save(changed, CHANGES)
        print(f"wrote {len(changed)} changed values to {CHANGES}")
    else:
        inputs = inputs_from_run()
        values = compute(inputs)
        FIXTURE.parent.mkdir(exist_ok=True)
        torch.save({"inputs": inputs, "values": values}, FIXTURE)
        size = sum(value.numel() for value in values.values())
        print(f"wrote {len(values)} golden values ({size} numbers) to {FIXTURE}")
