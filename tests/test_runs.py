"""Saving the composed matrices with the probe and reading the run back.

The round trip model -> ComposedMatrices artifact -> RunData.model(step) must
give back the same logits, for both backends: that is what lets every analysis
work offline from a run's saved matrices.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from tracklab import ExperimentTracker

from icl import (ComposedMatrices, Evaluator, RunData, TrainerArgs, build_model,
                 compute_derived_args, generate_icl_batch, log_artifacts, preprocess_batch)


def make_config(backend: str):
    cfg = OmegaConf.structured(TrainerArgs())
    cfg.model_args.update(vocab_size=24, seq_len=20, d_model=48, backend=backend)
    cfg.extra_args.experiment_name = f"roundtrip_{backend}"
    return compute_derived_args(cfg)


@pytest.mark.parametrize("backend", ["full", "reduced"])
def test_matrices_round_trip(tmp_path, backend):
    cfg = make_config(backend)
    torch.manual_seed(0)
    model, _, _ = build_model(cfg.model_args, cfg.optim_args, "cpu")
    model = model.double()
    V, L, K = cfg.model_args.vocab_size, cfg.model_args.seq_len, cfg.data_args.K
    test_batch, _ = preprocess_batch(generate_icl_batch(64, V, L, K))
    evaluator = Evaluator(artifacts=[ComposedMatrices()])

    exp = ExperimentTracker(cfg.extra_args.experiment_name, tmp_path)
    logits = {}
    with exp.start_run(cfg, artifacts=True) as run:
        for step in (0, 7):
            if step:                                   # different weights at the second step
                with torch.no_grad():
                    for p in model.parameters():
                        if p.requires_grad:
                            p.add_(0.1 * torch.randn_like(p))
            log_artifacts(run, evaluator.artifacts(model, test_batch, step=step), step)
            run.track_metric(step, loss=float(step))
            x = test_batch["sequence"][:, :-1]
            with torch.no_grad():
                logits[step] = model(x)
        run_id = run.run_id

    data = RunData(cfg.extra_args.experiment_name, run_id, base_dir=tmp_path)
    assert data.steps("matrices") == [0, 7]
    assert data.config.model_args.backend == backend and data.config.data_args.K == K
    assert list(data.metrics.index) == [0, 7] and data.metrics.loc[7, "loss"] == 7.0
    torch.testing.assert_close(data.matrices("first")["Q"], data.matrices(0)["Q"])
    with_profile = data.order_params(7, profile=True)
    torch.testing.assert_close(with_profile["M_profile"], data.matrices(7)["M"].double().diagonal(-1))
    assert with_profile["M_on"] == pytest.approx(with_profile["M_profile"].mean().item())
    for step, which in ((0, "first"), (7, "last")):
        loaded = data.model(which)
        assert loaded.matrices()["M"].dtype == torch.float64
        with torch.no_grad():
            torch.testing.assert_close(loaded(x), logits[step], rtol=1e-10, atol=1e-12)

    fresh = data.batch(5, seed=3)
    assert fresh["sequence"].shape == (5, L + 1)
    torch.testing.assert_close(fresh["sequence"], data.batch(5, seed=3)["sequence"])


def test_ansatz_init_keeps_the_order_parameters_of_the_draw():
    from icl import ansatz_matrices, build_model, measure_order_params
    cfg = make_config("reduced")
    cfg.model_args.update(init="sample", infinite_d=True)
    K = cfg.data_args.K
    torch.manual_seed(4)
    drawn, _, _ = build_model(cfg.model_args, cfg.optim_args, "cpu")
    cfg.model_args.ansatz_init = True
    torch.manual_seed(4)
    projected, _, message = build_model(cfg.model_args, cfg.optim_args, "cpu", K=K)
    assert "+ansatz" in message
    expected = measure_order_params(drawn.matrices(), K)
    assert measure_order_params(projected.matrices(), K) == pytest.approx(expected, rel=1e-5, abs=1e-7)
    for name, matrix in ansatz_matrices(expected, cfg.model_args.seq_len, cfg.model_args.vocab_size, K).items():
        torch.testing.assert_close(projected.matrices()[name].double(), matrix, rtol=1e-5, atol=1e-7)
    with pytest.raises(ValueError):
        build_model(cfg.model_args, cfg.optim_args, "cpu")                   # no K
    cfg.model_args.ansatz_keep_spread = "QG"
    torch.manual_seed(4)
    partial, _, _ = build_model(cfg.model_args, cfg.optim_args, "cpu", K=K)
    torch.testing.assert_close(partial.matrices()["M"], projected.matrices()["M"])
    for name in "QG":
        torch.testing.assert_close(partial.matrices()[name], drawn.matrices()[name])


def test_from_matrices_rejects_wrong_shapes():
    from icl import ReducedTransformer
    cfg = make_config("full")
    L, V = cfg.model_args.seq_len, cfg.model_args.vocab_size
    with pytest.raises(ValueError):
        ReducedTransformer.from_matrices({"M": torch.zeros(L + 1, L + 1), "Q": torch.zeros(V, V),
                                          "G": torch.zeros(V, V)}, cfg.model_args)
