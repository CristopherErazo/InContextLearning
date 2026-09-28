"""The reduced backend against the full model it rewrites.

The central check: in float64, SGD (with momentum and weight decay) on the full
model's (WQK1, WQK2, WOV2) and ReducedSGD on (m, q, g) follow the same trajectory,
so the reduced backend is the same experiment, not an approximation of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from icl import (Evaluator, LossMetric, MinimalTransformer, PerPositionOnOffLogits, ReducedSGD,
                 ReducedTransformer, TopKAccuracy, compute_loss, generate_icl_batch, preprocess_batch)


@dataclass
class Args:
    vocab_size: int = 24
    d_model: int = 48
    seq_len: int = 20
    lin_attn: bool = True
    beta: float = 0.25
    sigma_0: float = 1.0
    pred_mode: str = "next"
    dropout: float = 0.0
    mask1: str = "causal"
    mask2: str = "causal"


K = 4


def build_pair(**kw):
    torch.manual_seed(0)
    full = MinimalTransformer(Args(**kw)).double()
    full.initialize_model()
    return full, ReducedTransformer.from_full(full)


def composed(model):
    return {name: m for (name, _), m in model.get_composed_matrices().items()}


@pytest.mark.parametrize("kw", [{}, {"pred_mode": "last"}, {"mask1": "prev"}, {"mask2": "prev"}])
def test_forward_matches_full_model(kw):
    full, reduced = build_pair(**kw)
    x = generate_icl_batch(8, 24, 20, K)["sequence"][:, :-1]
    with torch.no_grad():
        torch.testing.assert_close(reduced(x), full(x), rtol=1e-10, atol=1e-12)
        out_full, out_red = full.full_output(x), reduced.full_output(x)
        for key in ("A1", "A2", "logits"):
            torch.testing.assert_close(out_red[key], out_full[key], rtol=1e-10, atol=1e-12)
    for name, m in composed(full).items():
        torch.testing.assert_close(composed(reduced)[name], m, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("momentum,weight_decay", [(0.0, 0.0), (0.9, 1e-2)])
def test_sgd_trajectory_matches_full_model(momentum, weight_decay):
    full, reduced = build_pair()
    alpha = 400.0                                    # eta_0; the full model's lr is alpha / d
    opt_full = torch.optim.SGD([p for p in full.parameters() if p.requires_grad],
                               lr=alpha / full.d_model, momentum=momentum, weight_decay=weight_decay)
    opt_red = ReducedSGD(reduced, lr=alpha, momentum=momentum, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    start = composed(full)
    for _ in range(40):
        batch = generate_icl_batch(16, 24, 20, K, stats=False)
        for model, opt in ((full, opt_full), (reduced, opt_red)):
            loss = compute_loss(model, batch, loss_fn, "cpu")
            opt.zero_grad()
            loss.backward()
            opt.step()
    end_full, end_red = composed(full), composed(reduced)
    for name in ("M", "Q", "G"):
        moved = (end_full[name] - start[name]).norm() / start[name].norm()
        assert moved > 0.05, f"{name} barely moved ({moved:.2g}); the check would be vacuous"
        torch.testing.assert_close(end_red[name], end_full[name], rtol=1e-9, atol=1e-9)


def test_sampled_init_has_the_full_models_law():
    """Second moments of (m, q, g) and of the Gram matrices, Monte Carlo over inits."""
    args, sigma, n = Args(vocab_size=8, seq_len=8, d_model=32, sigma_0=1.3), 1.3, 300

    def stats(model):
        off = ~torch.eye(8, dtype=torch.bool)
        return torch.stack([
            model.m.pow(2).mean(), model.q.pow(2).mean(), model.g.pow(2).mean(),
            model.kP.diagonal().mean(), model.kR.diagonal().mean(), model.kR[off].pow(2).mean(),
            model.kE[off].pow(2).mean(), (model.q * model.q.T).mean(),
        ]).detach()

    torch.manual_seed(1)
    from_full, sampled = [], []
    for _ in range(n):
        full = MinimalTransformer(args).double()
        full.initialize_model()
        from_full.append(stats(ReducedTransformer.from_full(full)))
        sampled.append(stats(ReducedTransformer.sample(args, 32, sigma, dtype=torch.float64)))
    a, b = torch.stack(from_full), torch.stack(sampled)
    se = (a.var(0) / n + b.var(0) / n).sqrt()
    z = (a.mean(0) - b.mean(0)) / se
    assert (z.abs() < 4).all(), f"moment z-scores {z.tolist()}"


def test_infinite_d():
    args = Args()
    model = ReducedTransformer.sample(args, math.inf, dtype=torch.float64)
    for name in ("kP", "kE", "kU", "kR"):
        n = getattr(model, name).size(0)
        torch.testing.assert_close(getattr(model, name), torch.eye(n, dtype=torch.float64))
    opt = ReducedSGD(model, lr=100.0, weight_decay=1.0)       # weight decay vanishes at d = inf
    loss = compute_loss(model, generate_icl_batch(8, 24, 20, K, stats=False), torch.nn.CrossEntropyLoss(), "cpu")
    loss.backward()
    opt.step()
    assert all(torch.isfinite(t).all() for t in model.normalized_matrices().values())


def test_chunked_evaluation_matches_one_pass():
    full, _ = build_pair()
    batch, _ = preprocess_batch(generate_icl_batch(64, 24, 20, K))
    probes = dict(scalars=[TopKAccuracy(1), LossMetric(), LossMetric("loss_ind", "ind")],
                  artifacts=[PerPositionOnOffLogits([0.5, 1.0])])
    one, chunked = Evaluator(**probes), Evaluator(**probes, chunk=7)
    s1, s2 = one.scalars(full, batch, step=0), chunked.scalars(full, batch, step=0)
    assert s1.keys() == s2.keys()
    for k in s1:
        assert s1[k] == pytest.approx(s2[k], rel=1e-12), k
    # the loss is the training loss: mean cross-entropy over every position
    ref = torch.nn.CrossEntropyLoss()(full(batch["sequence"][:, :-1]).flatten(0, 1),
                                      batch["sequence"][:, 1:].flatten())
    assert s1["loss"] == pytest.approx(ref.item(), rel=1e-12)
    a1 = one.artifacts(full, batch, step=0)[("hists", "logits")][0]
    a2 = chunked.artifacts(full, batch, step=0)[("hists", "logits")][0]
    assert a1["positions"] == a2["positions"] == [9, 19]
    for key in ("on", "off", "all"):
        for x, y in zip(a1[key], a2[key]):
            assert x.shape == y.shape and (abs(x - y) < 1e-12).all()


def test_logit_positions():
    probe = PerPositionOnOffLogits([1.0, 0.5, 0.75, 0.5])
    assert probe.fractions == [0.5, 0.75, 1.0]
    assert probe.positions(512) == [255, 383, 511]
    assert probe.positions(10) == [4, 7, 9]
    with pytest.raises(ValueError):
        PerPositionOnOffLogits([0.0])
