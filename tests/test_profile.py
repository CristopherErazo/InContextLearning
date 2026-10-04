"""The previous-token profile ("M_profile") in place of the scalar M_on.

The central check is the exactness test of test_theory.py with a random
profile: the model run on `ansatz_matrices` must match `ansatz_logits` on the
measured variables. A constant profile must give back the scalar ansatz, on
every source of variables (measured, sampled, mean).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from icl import (ORDER_PARAMS, EffectiveLoss, ReducedTransformer, ansatz_logits, ansatz_matrices,
                 generate_icl_batch, integrate, logit_table, mean_variables, measure_variables, sample_variables)


@dataclass
class Args:
    vocab_size: int = 24
    seq_len: int = 20
    lin_attn: bool = True
    beta: float = 0.25
    pred_mode: str = "next"
    dropout: float = 0.0
    mask1: str = "causal"
    mask2: str = "causal"


V, L, K = 24, 20, 4


def random_order_params(seed=0, profile=True):
    generator = torch.Generator().manual_seed(seed)
    order_params = {name: torch.randn(1, generator=generator, dtype=torch.float64).item() for name in ORDER_PARAMS}
    if profile:
        order_params["M_profile"] = torch.randn(L - 1, generator=generator, dtype=torch.float64)
    return order_params


@pytest.mark.parametrize("pred_mode", ["next", "last"])
def test_ansatz_logits_with_a_profile_are_exact(pred_mode):
    order_params = random_order_params(1)
    args = Args(pred_mode=pred_mode)
    model = ReducedTransformer.from_matrices(ansatz_matrices(order_params, L, V, K), args)
    batch = generate_icl_batch(256, V, L, K, generator=torch.Generator().manual_seed(2))
    mus = None if pred_mode == "next" else L
    model_logits = logit_table(model, batch, mus=mus, chunk=37)
    variables = measure_variables(batch, mus=mus, chunk=101)
    expected = ansatz_logits(variables, order_params, args.beta, L)
    torch.testing.assert_close(model_logits["logits"], expected, rtol=1e-12, atol=1e-12)


def test_ansatz_matrices_put_the_profile_on_the_sub_diagonal():
    order_params = random_order_params(3)
    M = ansatz_matrices(order_params, L, V, K)["M"]
    torch.testing.assert_close(M.diagonal(-1), order_params["M_profile"])
    with pytest.raises(ValueError):
        ansatz_matrices({"M_profile": torch.zeros(L)}, L, V, K)


def test_a_constant_profile_is_the_scalar_ansatz():
    scalar = random_order_params(4, profile=False)
    constant = dict(scalar, M_profile=torch.full((L - 1,), scalar["M_on"], dtype=torch.float64))
    batch = generate_icl_batch(128, V, L, K, generator=torch.Generator().manual_seed(5))
    sources = {
        "measured": measure_variables(batch),
        "sampled": sample_variables(torch.tensor([5, 12, 20]).repeat(50), 2, V, K, generator=torch.Generator().manual_seed(6)),
        "mean": mean_variables(torch.tensor([5, 12, 20]), torch.tensor([0, 1, 3]), V, K),
    }
    for name, variables in sources.items():
        torch.testing.assert_close(ansatz_logits(variables, constant, 0.25, L),
                                   ansatz_logits(variables, scalar, 0.25, L), rtol=1e-12, atol=1e-12, msg=name)


def test_measured_keys_match_the_counts():
    batch = generate_icl_batch(128, V, L, K, generator=torch.Generator().manual_seed(7))
    variables = measure_variables(batch, chunk=50)
    assert torch.equal((variables["N_keys"] > 0).sum(1).double(), variables["N"])
    assert torch.equal((variables["F_keys"] > 0).sum(1).double(), variables["F"])
    assert torch.equal((variables["U_keys"] > 0).sum(-1).double(), variables["U_bar"])
    keys = variables["N_keys"]
    assert ((keys == 0) | ((keys >= 1) & (keys <= (variables["mu"] - 2)[:, None]))).all()


def test_sampled_keys_and_the_profile_weighted_witness_sum():
    mu, ell, num_samples = 18, 3, 40000
    variables = sample_variables(mu, ell, V, K, num_samples=num_samples, generator=torch.Generator().manual_seed(8))
    assert torch.equal((variables["N_keys"] > 0).sum(1).double(), variables["N"])
    assert torch.equal((variables["F_keys"] > 0).sum(1).double(), variables["F"])
    assert torch.equal((variables["U_keys"] > 0).sum(-1).double(), variables["U_bar"])
    real = variables["N_keys"][variables["N_keys"] > 0]
    assert real.min() == 1 and real.max() == mu - 2                     # keys nu = 2..mu-1
    profile = torch.linspace(3.0, 0.5, L - 1, dtype=torch.float64) ** 2
    by_code_position = torch.cat([torch.zeros(1, dtype=torch.float64), profile])
    witness_sum = by_code_position[variables["N_keys"]].sum(1)
    window = profile[: mu - 2]                                          # the profile at the keys of the query
    mean, variance = ell * window.mean(), ell * window.var(unbiased=False)
    assert abs(witness_sum.mean() - mean) / (witness_sum.std() / num_samples ** 0.5) < 4.5
    centered = (witness_sum - witness_sum.mean()) ** 2
    assert abs(centered.mean() - variance) / (centered.std() / num_samples ** 0.5) < 4.5


@pytest.mark.parametrize("method", ["mean", "mc"])
def test_effective_loss_gradient_with_respect_to_the_profile(method):
    loss = EffectiveLoss(V, L, K, 0.25, method=method, mus=[8, 14, 20], num_samples=8)
    base = random_order_params(9)
    base.update({"M_off": 0.2, "Q_on": 6.0, "G_on": 4.0})

    def as_function(profile):
        return loss(dict(base, M_profile=profile))

    profile = (2.0 + base["M_profile"]).clone().requires_grad_(True)
    assert torch.autograd.gradcheck(as_function, (profile,))
    value, gradient = loss.value_and_grad(dict(base, M_profile=profile.detach()))
    assert gradient["M_on"] == 0.0 and gradient["M_profile"].shape == (L - 1,)
    torch.testing.assert_close(gradient["M_profile"], torch.autograd.grad(as_function(profile), profile)[0])


def test_flow_of_the_profile():
    target, curvature = torch.linspace(1.0, 3.0, L - 1, dtype=torch.float64), 0.5

    def loss(order_params):
        return curvature * ((order_params["M_profile"] - target) ** 2).sum() + (order_params["Q_on"] - 1.0) ** 2

    rates = torch.linspace(0.01, 0.1, L - 1, dtype=torch.float64)
    start = {"M_profile": torch.zeros(L - 1, dtype=torch.float64), "Q_on": 0.0}
    flow = integrate(loss, start, {"M_profile": rates, "Q_on": 0.05}, steps=50, record_steps=[0, 25, 50],
                     rtol=1e-10, atol=1e-12)
    for step in (25, 50):
        expected = target * (1 - torch.exp(-2 * curvature * rates * step))
        assert flow.loc[step, "M_profile"] == pytest.approx(expected.numpy(), rel=1e-6)
        assert flow.loc[step, "M_on"] == pytest.approx(float(np.mean(flow.loc[step, "M_profile"])))
        assert flow.loc[step, "Q_on"] == pytest.approx(1 - np.exp(-2 * 0.05 * step), rel=1e-6)
    assert (flow["G_on"] == 0.0).all()                                   # not moving
    frozen = integrate(loss, dict(start, M_profile=target.clone()), {"Q_on": 0.05}, steps=10, record_steps=[0, 10])
    assert frozen.loc[10, "M_profile"] == pytest.approx(target.numpy())  # a fixed profile stays put


def test_flow_inside_a_profile_family():
    x = torch.arange(2, L + 1, dtype=torch.float64) / L
    basis = torch.stack([torch.ones_like(x), x], dim=1)                 # the linear family m = theta_0 + theta_1 x
    target, curvature, rate = torch.cos(3 * x), 0.5, 0.02               # not in the family

    def loss(order_params):
        return curvature * ((order_params["M_profile"] - target) ** 2).sum()

    theta0 = torch.tensor([1.0, -1.0], dtype=torch.float64)
    flow = integrate(loss, {}, {"M_profile": rate}, steps=40, record_steps=[0, 20, 40], rtol=1e-10, atol=1e-12,
                     profile_family=(lambda theta: basis @ theta, theta0))
    best = torch.linalg.lstsq(basis, target[:, None]).solution[:, 0]    # the projected flow relaxes to the fit
    for step in (20, 40):
        expected = best + np.exp(-2 * curvature * rate * step) * (theta0 - best)
        assert flow.loc[step, "M_profile_params"] == pytest.approx(expected.numpy(), rel=1e-6)
        assert flow.loc[step, "M_profile"] == pytest.approx((basis @ expected).numpy(), rel=1e-6)
    with pytest.raises(ValueError):
        integrate(loss, {"M_profile": target}, {"M_profile": rate}, steps=1, profile_family=(lambda t: basis @ t, theta0))


def test_the_constant_family_is_the_scalar_flow():
    loss = EffectiveLoss(V, L, K, 0.25, method="mean", mus=[8, 14, 20])
    start = {"M_on": 0.5, "Q_on": 2.0, "G_on": 1.5, "Q_T": -0.3}
    rate = 0.4
    kwargs = dict(steps=30, record_steps=[0, 15, 30], rtol=1e-10, atol=1e-12)
    scalar = integrate(loss, start, {"M_on": rate / (L - 1), "Q_on": 0.1}, **kwargs)
    family = integrate(loss, {name: value for name, value in start.items() if name != "M_on"},
                       {"M_profile": rate, "Q_on": 0.1}, **kwargs,
                       profile_family=(lambda theta: theta.expand(L - 1), torch.tensor([0.5])))
    for name in ("M_on", "Q_on", "loss"):
        assert family[name].to_numpy() == pytest.approx(scalar[name].to_numpy(), rel=1e-7)
    assert family.loc[30, "M_profile_params"][0] == pytest.approx(scalar.loc[30, "M_on"], rel=1e-7)
