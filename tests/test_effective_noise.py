"""The effective loss with the logit noise of the variance ansatz: without variances
it is the signal-only loss; the noise raises it; the two methods agree on its cost;
the closure is the Gaussian integral it stands for; gradients; the non-trigger part
against the model on noisy ansatz matrices. (The trigger part against the model
needs ~100 networks: one network's trigger loss scatters by ~0.1 nats here, since
the K rows of Q at the trigger queries are quenched. Averaged over 128 networks the
noise costs 0.0465 +- 0.0010 of loss, 0.122 +- 0.007 at the trigger queries; the
effective loss gives 0.049 and 0.13-0.134.)"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from icl import (EffectiveLoss, ReducedTransformer, ansatz_matrices, at_level, closure_cross_entropy,
                 generate_icl_batch, logit_blocks)


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
MEANS = {"M_on": 8.0, "M_off": 0.1, "Q_on": 12.0, "Q_T": -0.5, "G_on": 6.0, "G_T": -1.0}
SPREAD = {"var_M_on": 4.0, "var_M_off": 0.04, "var_Q_on": 9.0, "var_Q_T": 1.0, "var_Q_N": 1.0, "var_G_on": 4.0,
          "var_G_T": 1.0, "var_G_N": 1.0}
MUS = range(8, L + 1, 8)


@pytest.mark.parametrize("method", ["mean", "mc"])
def test_without_variances_the_loss_is_the_signal_only_one(method):
    loss = EffectiveLoss(V, L, K, BETA, method=method, mus=MUS, num_samples=32)
    assert torch.equal(loss(at_level({**MEANS, **SPREAD}, "S3")), loss(MEANS))
    assert loss.breakdown(MEANS)["loss_non_trigg"] == math.log(V)
    # variances present but 0: the noise paths reduce to the same loss
    zero = {name: 0.0 for name in SPREAD}
    value, gradient = loss.value_and_grad(MEANS)
    value_zero, gradient_zero = loss.value_and_grad({**MEANS, **zero})
    assert value_zero == pytest.approx(value, rel=1e-12)
    for name in MEANS:
        assert gradient_zero[name] == pytest.approx(gradient[name], rel=1e-9, abs=1e-12), name
    assert set(gradient_zero) == {*MEANS, *zero} and all(math.isfinite(gradient_zero[name]) for name in zero)


@pytest.mark.parametrize("method", ["mean", "mc"])
def test_noise_raises_the_loss(method):
    loss = EffectiveLoss(V, L, K, BETA, method=method, mus=MUS, num_samples=32)
    quiet, noisy = loss.breakdown(MEANS), loss.breakdown({**MEANS, **SPREAD})
    for part in ("loss", "loss_trigg", "loss_non_trigg"):
        assert noisy[part] > quiet[part], part
    pooled = loss.breakdown({**MEANS, "var_M": 1.0, "var_Q": 1.0, "var_G": 1.0})        # S1: one per matrix
    assert pooled["loss"] > quiet["loss"]


def test_the_methods_agree_on_the_cost_of_the_noise():
    costs = {}
    for method in ("mean", "mc"):
        loss = EffectiveLoss(V, L, K, BETA, method=method, mus=MUS, num_samples=128)
        costs[method] = {part: loss.breakdown({**MEANS, **SPREAD})[part] - loss.breakdown(MEANS)[part]
                         for part in ("loss", "loss_trigg", "loss_non_trigg")}
    for part, cost in costs["mean"].items():
        assert cost == pytest.approx(costs["mc"][part], rel=0.05), part


def test_the_closure_is_the_gaussian_integral():
    """The closure of the target and the common trigger part, against a direct Monte
    Carlo of the same two-dimensional Gaussian integral."""
    generator = torch.Generator().manual_seed(0)
    num_rows, small_V, small_K = 3, 16, 3
    logits = torch.randn(num_rows, small_V, generator=generator, dtype=torch.float64)
    logits[:, :small_K] = logits[:, :1]                                  # the K triggers share their mean
    variances = {"target": torch.tensor([0.5, 2.0, 0.1], dtype=torch.float64),
                 "trigger_common": torch.tensor([0.3, 1.0, 0.4], dtype=torch.float64),
                 "target_trigger": torch.tensor([0.2, -0.8, 0.0], dtype=torch.float64),
                 "trigger_individual": torch.tensor([0.2, 0.5, 1.0], dtype=torch.float64),
                 "non_target": torch.rand(num_rows, small_V - small_K - 1, generator=generator, dtype=torch.float64)}
    closure = closure_cross_entropy(logits, variances, small_K)

    blocks = logit_blocks(small_K, small_V)
    covariance = torch.stack([torch.stack([variances["target"], variances["target_trigger"]], -1),
                              torch.stack([variances["target_trigger"], variances["trigger_common"]], -1)], -2)
    normals = torch.randn(400_000, num_rows, 2, generator=generator, dtype=torch.float64)
    xi = torch.einsum("rij,srj->sri", torch.linalg.cholesky(covariance), normals)
    target = logits[:, -1] + xi[..., 0]
    triggers = math.log(small_K) + logits[:, 0] + variances["trigger_individual"] / 2 + xi[..., 1]
    non_targets = torch.logsumexp(logits[:, blocks["non_target"]] + variances["non_target"] / 2, dim=1).expand_as(target)
    direct = (torch.logsumexp(torch.stack([target, triggers, non_targets], -1), -1) - target).mean(0)
    torch.testing.assert_close(closure, direct, rtol=2e-3, atol=0)

    zero = {name: torch.zeros_like(value) for name, value in variances.items()}
    torch.testing.assert_close(closure_cross_entropy(logits, zero, small_K),
                               torch.logsumexp(logits, 1) - logits[:, -1], rtol=1e-12, atol=0)


@pytest.mark.parametrize("method", ["mean", "mc"])
def test_gradient(method):
    small = dict(V=16, L=24, K=3)
    loss = EffectiveLoss(**small, beta=BETA, method=method, mus=[6, 16, 24], num_samples=4, noise_samples=4)
    names = [*MEANS, "var_M_on", "var_Q_on", "var_Q_T", "var_G_on", "var_G_T", "var_G_N", "var_Q"]
    values = [*MEANS.values(), 1.0, 2.0, 0.5, 1.0, 0.3, 0.4, 0.6]
    inputs = [torch.tensor(value / 4, dtype=torch.float64, requires_grad=True) for value in values]
    assert torch.autograd.gradcheck(lambda *params: loss(dict(zip(names, params))), inputs, eps=1e-6, atol=1e-6)


def test_non_trigger_loss_matches_the_model():
    """The non-trigger part averages over many rows of Q, so a few networks suffice:
    the cross-entropy averaged over the uniform next token, at the non-trigger queries."""
    op = {**MEANS, **SPREAD}
    batch = generate_icl_batch(1024, V, L, K, generator=torch.Generator().manual_seed(1))
    inputs = batch["sequence"][:, :-1]
    measured = []
    for seed in range(8):
        model = ReducedTransformer.from_matrices(ansatz_matrices(op, L, V, K, noise=torch.Generator().manual_seed(seed)),
                                                 Args())
        with torch.no_grad():
            logits = model(inputs)
        measured.append((torch.logsumexp(logits, -1) - logits.mean(-1))[inputs >= K].mean() - math.log(V))
    effective = EffectiveLoss(V, L, K, BETA).non_trigger_loss(op) - math.log(V)
    assert effective.item() == pytest.approx(torch.stack(measured).mean().item(), rel=0.1)
