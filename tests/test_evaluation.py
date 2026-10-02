"""The scalar probes against the same quantities computed by hand from the
model's logits: the position subsets of LossMetric (all / trigg / non_trigg,
and ell), and the in-context accuracy (trigger queries with ell >= 1 only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from icl import (Evaluator, LossMetric, MinimalTransformer, TargetProbMass, TopKAccuracy,
                 generate_icl_batch, preprocess_batch)
from icl.data import occurrence_counts


@dataclass
class Args:
    vocab_size: int = 24
    d_model: int = 32
    seq_len: int = 20
    lin_attn: bool = True
    beta: float = 4.0
    sigma_0: float = 1.0
    pred_mode: str = "next"
    dropout: float = 0.0
    mask1: str = "causal"
    mask2: str = "causal"


V, L, K = 24, 20, 4


def setup(pred_mode="next"):
    torch.manual_seed(0)
    model = MinimalTransformer(Args(pred_mode=pred_mode)).double()
    model.initialize_model()
    test_batch, _ = preprocess_batch(generate_icl_batch(128, V, L, K, generator=torch.Generator().manual_seed(1)))
    inputs = test_batch["sequence"][:, :-1]
    targets = test_batch["sequence"][:, 1:]
    with torch.no_grad():
        logits = model(inputs)
    if pred_mode == "last":
        inputs_ell = occurrence_counts(inputs)[:, -1:] - 1
        is_trigger, targets = inputs[:, -1:] < K, targets[:, -1:]
    else:
        inputs_ell, is_trigger = occurrence_counts(inputs) - 1, inputs < K
    token_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)
    return model, test_batch, logits, targets, token_loss, is_trigger, inputs_ell


@pytest.mark.parametrize("pred_mode", ["next", "last"])
def test_loss_subsets(pred_mode):
    model, test_batch, _, _, token_loss, is_trigger, ell = setup(pred_mode)
    probes = [LossMetric(), LossMetric(positions="trigg"), LossMetric(positions="non_trigg"),
              LossMetric(positions="trigg", ell=0), LossMetric(positions="trigg", ell=[1, 2]),
              LossMetric(positions="trigg", ell=">=1")]
    metrics = Evaluator(scalars=probes, chunk=17).scalars(model, test_batch, step=0)
    assert list(metrics) == ["loss", "loss_trigg", "loss_non_trigg", "loss_trigg_ell0",
                             "loss_trigg_ell1_2", "loss_trigg_ell_ge1"]
    expected = {
        "loss": token_loss.mean(),
        "loss_trigg": token_loss[is_trigger].mean(),
        "loss_non_trigg": token_loss[~is_trigger].mean(),
        "loss_trigg_ell0": token_loss[is_trigger & (ell == 0)].mean(),
        "loss_trigg_ell1_2": token_loss[is_trigger & ((ell == 1) | (ell == 2))].mean(),
        "loss_trigg_ell_ge1": token_loss[is_trigger & (ell >= 1)].mean(),
    }
    for name, value in expected.items():
        if math.isnan(value.item()):
            assert math.isnan(metrics[name]), name
        else:
            assert metrics[name] == pytest.approx(value.item(), rel=1e-12), name
    if pred_mode == "next":
        # the population loss splits into the trigger and the non-trigger positions
        frac = is_trigger.double().mean().item()
        assert metrics["loss"] == pytest.approx(frac * metrics["loss_trigg"] + (1 - frac) * metrics["loss_non_trigg"])


def test_ell_absent_from_the_batch_gives_nan():
    model, test_batch, *_ = setup()
    probes = [LossMetric(positions="trigg", ell=50), LossMetric(positions="trigg", ell=[40, 41]),
              LossMetric(positions="trigg", ell=">=40")]
    metrics = Evaluator(scalars=probes).scalars(model, test_batch, step=0)
    assert all(math.isnan(value) for value in metrics.values())


def test_loss_metric_rejects_bad_options():
    for bad in (dict(positions="ind"), dict(ell=0), dict(positions="non_trigg", ell=1),
                dict(positions="trigg", ell=1.5), dict(positions="trigg", ell=True),
                dict(positions="trigg", ell="> 1"), dict(positions="trigg", ell=[])):
        with pytest.raises(ValueError):
            LossMetric(**bad)
    assert LossMetric("my_loss", positions="trigg", ell=0).name == "my_loss"


def test_in_context_accuracy_uses_ell_at_least_one():
    model, test_batch, logits, targets, _, is_trigger, ell = setup()
    metrics = Evaluator(scalars=[TopKAccuracy(1), TopKAccuracy(3), TargetProbMass()]).scalars(model, test_batch, step=0)
    in_context = is_trigger & (ell >= 1)
    rows, target = logits[in_context], targets[in_context]
    for k in (1, 3):
        hits = (rows.topk(k, dim=-1).indices == target[:, None]).any(-1).double().mean()
        assert metrics[f"top{k}_accuracy"] == pytest.approx(hits.item(), rel=1e-6)   # the probe averages in float32
    probability = torch.softmax(rows, -1).gather(1, target[:, None]).mean()
    assert metrics["target_prob"] == pytest.approx(probability.item(), rel=1e-12)


def test_accuracy_without_in_context_queries_is_nan():
    torch.manual_seed(0)
    model = MinimalTransformer(Args()).double()
    model.initialize_model()
    # every trigger appears once: no query has ell >= 1
    sequence = torch.tensor([[0, 5, 9, 1, 6, 10, 2, 7, 11, 3, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])
    test_batch = {"sequence": sequence, "output_set": torch.tensor([[5, 6, 7, 8]]), "K": K, "V": V}
    metrics = Evaluator(scalars=[TopKAccuracy(1), TargetProbMass(), LossMetric(positions="trigg", ell=0)]).scalars(
        model, test_batch, step=0)
    assert math.isnan(metrics["top1_accuracy"]) and math.isnan(metrics["target_prob"])
    assert math.isfinite(metrics["loss_trigg_ell0"])
