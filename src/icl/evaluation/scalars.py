"""Scalar probes: each maps an EvalContext to a float (or a dict of floats)
that is logged as a metric. Add a new one here and pass it to
`Evaluator(scalars=[...])`.
"""
from __future__ import annotations

from typing import Literal

import torch

from .evaluator import EvalContext

MaskName = Literal["all", "ind", "no_ind"]


class LossMetric:
    """Cross-entropy on a subset of positions: every position, only
    induction-possible ones, or only the rest."""

    def __init__(self, name: str = "loss", mask: MaskName = "all"):
        if mask not in ("all", "ind", "no_ind"):
            raise ValueError(f"mask must be 'all', 'ind' or 'no_ind', got {mask!r}")
        self.name, self.mask = name, mask

    def __call__(self, ctx: EvalContext) -> float:
        if ctx.loss_fn is None:
            raise RuntimeError("LossMetric needs Evaluator(loss_fn=...)")
        mask = {"all": ctx.all, "ind": ctx.ind_possible, "no_ind": ctx.ind_not_possible}[self.mask]
        return ctx.loss_fn(ctx.logits[mask], ctx.target[mask]).item()


class TopKAccuracy:
    """Fraction of induction-possible positions whose target is in the top-k logits."""

    def __init__(self, k: int = 1):
        self.k, self.name = k, f"top{k}_accuracy"

    def __call__(self, ctx: EvalContext) -> float:
        topk = ctx.logits_ind.topk(self.k, dim=-1).indices              # (N, k)
        return (topk == ctx.target_ind[:, None]).any(dim=-1).float().mean().item()


class TargetProbMass:
    """Mean softmax probability of the target at induction-possible positions."""

    name = "target_prob"

    def __call__(self, ctx: EvalContext) -> float:
        probs = torch.softmax(ctx.logits_ind, dim=-1)                    # (N, V)
        return probs.gather(1, ctx.target_ind[:, None]).mean().item()


class LogitStatistics:
    """Mean / variance of on- and off-target logits at induction-possible
    positions, plus their covariance."""

    name = "logit_statistics"
    names = ("on_logit_mean", "on_logit_var", "off_logit_mean", "off_logit_var", "on_off_covariance")

    def __call__(self, ctx: EvalContext) -> dict[str, float]:
        on, off = ctx.on_target_logits, ctx.off_target_logits           # (N,), (N, V-1)
        mean_on, mean_off = on.mean(), off.mean()
        cov = ((on[:, None] - mean_on) * (off - mean_off)).mean()
        return {
            "on_logit_mean": mean_on.item(),
            "on_logit_var": on.var().item(),
            "off_logit_mean": mean_off.item(),
            "off_logit_var": off.var().item(),
            "on_off_covariance": cov.item(),
        }
