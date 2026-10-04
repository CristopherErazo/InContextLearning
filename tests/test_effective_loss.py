"""The effective population loss: exact limits, its gradient, and its agreement
with the measured loss of a model whose matrices are exactly in the ansatz
(the only differences left are the approximations of the variables' law).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
from omegaconf import OmegaConf

from icl import (ORDER_PARAMS, EffectiveLoss, Evaluator, LossMetric, ReducedTransformer, TrainerArgs,
                 ansatz_matrices, compute_derived_args, generate_icl_batch, preprocess_batch, trigger_loss)


@dataclass
class Args:
    vocab_size: int = 48
    seq_len: int = 96
    lin_attn: bool = True
    beta: float = 0.25
    pred_mode: str = "next"
    dropout: float = 0.0
    mask1: str = "causal"
    mask2: str = "causal"


V, L, K, BETA = 48, 96, 9, 0.25
FULL_ANSATZ = {"M_on": 8.0, "M_off": 0.1, "Q_on": 12.0, "Q_T": -0.5, "G_on": 6.0, "G_T": -1.0}


@pytest.mark.parametrize("method", ["mean", "mc"])
def test_zero_order_params_give_log_v(method):
    loss = EffectiveLoss(V, L, K, BETA, method=method, mus=[10, 50, 96], num_samples=16)
    assert loss({}).item() == pytest.approx(math.log(V), rel=1e-12)
    parts = loss.breakdown({})
    assert parts["loss_trigg"] == pytest.approx(math.log(V)) and parts["loss_non_trigg"] == math.log(V)


def test_cluster_probabilities():
    loss = EffectiveLoss(V, L, K, BETA, mus=[1, 48, 96], ell_tol=1e-9)
    clusters = loss.clusters
    for mu in (1, 48, 96):
        in_mu = clusters["mu"] == mu
        assert clusters["probability"][in_mu].sum().item() == pytest.approx(1.0, rel=1e-12)
        mean_ell = (clusters["probability"][in_mu] * clusters["ell"][in_mu]).sum().item()
        assert mean_ell == pytest.approx(mu / (V + K), rel=1e-6)          # ell ~ Poisson(p_T mu)


@pytest.mark.parametrize("order_params", [{"M_on": 8.0, "Q_on": 12.0, "G_on": 6.0}, FULL_ANSATZ])
def test_effective_loss_matches_the_exact_ansatz_model(order_params):
    model = ReducedTransformer.from_matrices(ansatz_matrices(order_params, L, V, K), Args())
    test_batch, _ = preprocess_batch(generate_icl_batch(16384, V, L, K, generator=torch.Generator().manual_seed(0)))
    probes = [LossMetric(), LossMetric(positions="trigg", ell=0), LossMetric(positions="trigg", ell=">=1")]
    measured = Evaluator(scalars=probes, chunk=4096).scalars(model, test_batch, step=0)
    for method in ("mean", "mc"):
        parts = EffectiveLoss(V, L, K, BETA, method=method).breakdown(order_params)
        assert parts["loss"] == pytest.approx(measured["loss"], rel=5e-3), method
        assert trigger_loss(parts["clusters"], 0) == pytest.approx(measured["loss_trigg_ell0"], rel=1e-2), method
        assert trigger_loss(parts["clusters"], ">=1") == pytest.approx(measured["loss_trigg_ell_ge1"], rel=3e-2), method


@pytest.mark.parametrize("method", ["mean", "mc"])
def test_gradient(method):
    loss = EffectiveLoss(V, L, K, BETA, method=method, mus=[20, 60, 96], num_samples=8)
    names = list(ORDER_PARAMS)

    def as_function(vector):
        return loss(dict(zip(names, vector)))

    point = torch.tensor([FULL_ANSATZ[name] for name in names], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(as_function, (point,))
    value, gradient = loss.value_and_grad(FULL_ANSATZ)
    assert list(gradient) == names and value == pytest.approx(as_function(point).item())
    autograd = torch.autograd.grad(as_function(point), point)[0]
    assert [gradient[name] for name in names] == pytest.approx(autograd.tolist(), rel=1e-12)


def test_mc_is_reproducible_and_the_options_are_checked():
    first = EffectiveLoss(V, L, K, BETA, method="mc", mus=[50], num_samples=32, seed=3)
    second = EffectiveLoss(V, L, K, BETA, method="mc", mus=[50], num_samples=32, seed=3)
    assert first(FULL_ANSATZ).item() == second(FULL_ANSATZ).item()
    for bad in (dict(method="exact"), dict(counts="binomial"), dict(mus=[0]), dict(mus=[L + 1])):
        with pytest.raises(ValueError):
            EffectiveLoss(V, L, K, BETA, **bad)


def test_from_config():
    config = OmegaConf.structured(TrainerArgs())
    config.model_args.update(vocab_size=V, seq_len=L, beta=BETA)
    config = compute_derived_args(config)
    loss = EffectiveLoss.from_config(config, mus=[L])
    assert (loss.V, loss.L, loss.K, loss.beta) == (V, L, config.data_args.K, BETA)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("counts", ["poisson", "multinomial"])
def test_runs_on_the_gpu(counts):
    on_cpu = EffectiveLoss(V, L, K, BETA, method="mc", mus=[48, 96], counts=counts, num_samples=2048)
    on_gpu = EffectiveLoss(V, L, K, BETA, method="mc", mus=[48, 96], counts=counts, num_samples=2048, device="cuda")
    value, gradient = on_gpu.value_and_grad(FULL_ANSATZ)
    assert value == pytest.approx(on_cpu(FULL_ANSATZ).item(), rel=1e-2)        # different draws, same law
    assert all(math.isfinite(g) for g in gradient.values())


def test_asymptotic_loss_limits():
    from icl import asymptotic_loss
    V, K = 128, 25
    trigger = K / (V + K)
    assert asymptotic_loss(V, 10**7, K) == pytest.approx((1 - trigger) * math.log(V), rel=1e-5)   # every trigger solvable
    assert asymptotic_loss(V, 1e-6, K) == pytest.approx(math.log(V) + trigger * math.log(1 - K / V), rel=1e-5)
    assert asymptotic_loss(V, 64, K) > asymptotic_loss(V, 1024, K)


def test_learning_time_interpolates_the_first_crossing():
    import pandas as pd
    from icl import asymptotic_loss, learning_time
    V, L, K = 128, 256, 25
    gap = math.log(V) - asymptotic_loss(V, L, K)
    progress = pd.Series([0.0, 0.1, 0.3, 0.15, 0.9], index=[0, 10, 20, 30, 40])
    loss = math.log(V) - gap * progress
    assert learning_time(loss, V, L, K, fraction=0.2) == pytest.approx(15.0)
    assert math.isnan(learning_time(loss, V, L, K, fraction=0.95))
