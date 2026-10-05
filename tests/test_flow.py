"""The gradient-flow integrator: the closed-form solution of a quadratic loss,
freezing by omission, and descent of the effective loss; the multiplicative flow of
the variances, and `exact_rates` against the reduced model's SGD step."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from icl import (ORDER_PARAMS, PROFILE, EffectiveLoss, ReducedSGD, TrainerArgs, ansatz_matrices, build_model,
                 compute_derived_args, compute_loss, exact_rates, generate_icl_batch, integrate, measure_order_params,
                 support_sizes, variance_sizes)


def quadratic_loss(curvature, minimum):
    def loss(order_params):
        return sum(curvature[name] * (torch.as_tensor(order_params.get(name, 0.0), dtype=torch.float64)
                                      - minimum[name]) ** 2 for name in curvature)
    return loss


def test_quadratic_flow_matches_the_closed_form():
    curvature = {"M_on": 1.0, "Q_on": 0.5, "G_on": 2.0}
    minimum = {"M_on": 3.0, "Q_on": -1.0, "G_on": 0.5}
    rates = {"M_on": 0.01, "Q_on": 0.05}                          # G_on is frozen
    start = {"M_on": 0.0, "Q_on": 2.0, "G_on": 1.0}
    flow = integrate(quadratic_loss(curvature, minimum), start, rates, steps=500,
                     record_steps=[0, 100, 250, 500], rtol=1e-10, atol=1e-12)
    assert list(flow.index) == [0, 100, 250, 500]
    assert list(flow.columns) == [*ORDER_PARAMS, "loss"]
    for name, rate in rates.items():
        decay = np.exp(-2 * curvature[name] * rate * flow.index.to_numpy())
        expected = minimum[name] + (start[name] - minimum[name]) * decay
        assert flow[name].to_numpy() == pytest.approx(expected, rel=1e-7, abs=1e-9)
    assert (flow["G_on"] == 1.0).all()                             # not in rates: stays put
    assert (flow["M_off"] == 0.0).all()                            # not given: 0, and frozen
    assert flow["loss"].is_monotonic_decreasing


def test_effective_loss_decreases_along_the_flow():
    loss = EffectiveLoss(48, 96, 9, 0.25, mus=[24, 48, 72, 96])
    start = {"M_on": 2.0, "M_off": 0.01, "Q_on": 3.0, "Q_T": 0.1, "G_on": 2.0, "G_T": 0.1}
    rates = {name: 50.0 for name in ("M_on", "Q_on", "G_on")}     # the signal-only flow
    flow = integrate(loss, start, rates, steps=200, record_steps=np.linspace(0, 200, 11))
    assert flow["loss"].is_monotonic_decreasing
    assert flow["loss"].iloc[-1] < flow["loss"].iloc[0]
    for frozen in ("M_off", "Q_T", "G_T"):
        assert (flow[frozen] == start[frozen]).all()


def test_zero_is_a_fixed_point_and_options_are_checked():
    loss = EffectiveLoss(48, 96, 9, 0.25, mus=[96])
    flow = integrate(loss, {}, {"M_on": 1.0, "Q_on": 1.0, "G_on": 1.0}, steps=10, record_steps=[0, 10])
    assert (flow[["M_on", "Q_on", "G_on"]] == 0.0).all().all()
    assert flow["loss"].iloc[-1] == pytest.approx(math.log(48))
    with pytest.raises(ValueError):
        integrate(loss, {}, {"M_typo": 1.0}, steps=1)
    with pytest.raises(ValueError):
        integrate(loss, {}, {}, steps=1)


def test_variances_flow_multiplicatively():
    """For a loss linear in the variances, var(t) = var(0) exp(-rate c t); a variance at
    0 stays there, and one without a rate is handed to the loss unchanged."""
    slopes = {"var_Q": 2.0, "var_G": 1.0, "var_M": 3.0}

    def loss(order_params):
        return sum(slope * torch.as_tensor(order_params.get(name, 0.0), dtype=torch.float64)
                   for name, slope in slopes.items()) + (torch.as_tensor(order_params.get("M_on", 0.0)) - 1) ** 2

    start = {"M_on": 0.0, "var_Q": 0.5, "var_G": 0.0, "var_M": 0.2}
    rates = {"M_on": 0.01, "var_Q": 0.003, "var_G": 0.01}
    flow = integrate(loss, start, rates, steps=400, record_steps=[0, 100, 400], rtol=1e-10, atol=1e-14)
    assert list(flow.columns) == [*ORDER_PARAMS, "var_M", "var_Q", "var_G", "loss"]
    expected = 0.5 * np.exp(-rates["var_Q"] * slopes["var_Q"] * flow.index.to_numpy())
    assert flow["var_Q"].to_numpy() == pytest.approx(expected, rel=1e-7)
    assert (flow["var_G"] == 0.0).all() and (flow["var_M"] == 0.2).all()
    assert flow["loss"].iloc[0] == pytest.approx(1 + 2.0 * 0.5 + 3.0 * 0.2)
    with pytest.raises(ValueError):
        integrate(loss, start, {"var_M_on_raw": 1.0}, steps=1)


def make_config(sigma_0: float):
    config = OmegaConf.structured(TrainerArgs())
    config.model_args.update(vocab_size=24, seq_len=20, backend="reduced", init="sample", infinite_d=True,
                             sigma_0=sigma_0)
    config.optim_args.alpha_lr = 300.0
    return compute_derived_args(config)


def test_exact_rates():
    config = make_config(sigma_0=0.5)
    V, L, K, eta_0 = config.model_args.vocab_size, config.model_args.seq_len, config.data_args.K, 300.0
    sizes = {**support_sizes(L, V, K), **variance_sizes(L, V, K)}
    rates = exact_rates(config, [*ORDER_PARAMS, PROFILE, "var_M_on", "var_Q_T", "var_G_N", "var_Q"])
    assert rates["M_on"] == eta_0 / (L - 1) and rates[PROFILE] == eta_0
    assert rates["Q_on"] == eta_0 * 0.25 / K and rates["G_T"] == eta_0 / (K * V)
    assert rates["var_Q_T"] == 4 * eta_0 * 0.25 / (K * (V - 1)) and rates["var_G_N"] == 4 * eta_0 / sizes["var_G_N"]
    assert rates["var_Q"] == 4 * eta_0 * 0.25 / V ** 2 and rates["var_M_on"] == 4 * eta_0 / (L - 1)
    # sigma_0 = 1: alpha_lr / (support size), the rates the flows used before
    assert exact_rates(make_config(1.0), ORDER_PARAMS) == {name: eta_0 / size for name, size in
                                                           support_sizes(L, V, K).items()}
    with pytest.raises(ValueError):
        exact_rates(config, ["var_M_on_raw"])
    config.optim_args.momentum = 0.9
    with pytest.raises(ValueError):
        exact_rates(config, ["M_on"])


def test_exact_rates_are_the_reduced_sgd_step():
    """One SGD step of the reduced model at d = inf moves each block mean by -rate times
    the batch gradient with respect to it (the sum of its entries' gradients)."""
    config = make_config(sigma_0=0.5)
    V, L, K = config.model_args.vocab_size, config.model_args.seq_len, config.data_args.K
    torch.manual_seed(0)
    model = build_model(config.model_args, config.optim_args, "cpu")[0].double()
    optimizer = ReducedSGD(model, lr=config.optim_args.alpha_lr)                   # as build_model, in float64
    means = {"M_on": 2.0, "M_off": 0.1, "Q_on": 3.0, "Q_T": 0.2, "G_on": 2.0, "G_T": -0.5}
    matrices = ansatz_matrices(means, L, V, K)
    model._load({"m": matrices["M"], "q": matrices["Q"], "g": matrices["G"]})
    batch = generate_icl_batch(64, V, L, K, stats=False, generator=torch.Generator().manual_seed(2))
    compute_loss(model, batch, torch.nn.CrossEntropyLoss(), "cpu").backward()
    gradients = {"M": model.m.grad, "Q": model.q.grad, "G": model.g.grad}
    before = measure_order_params(model.matrices(), K)
    optimizer.step()
    after = measure_order_params(model.matrices(), K)
    rates = exact_rates(config, ORDER_PARAMS)
    for name, (matrix, support) in ORDER_PARAMS.items():
        gradient = gradients[matrix][support(L, V, K)].sum().item()
        assert after[name] - before[name] == pytest.approx(-rates[name] * gradient, rel=1e-9), name
