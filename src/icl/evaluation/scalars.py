"""Scalar probes: each maps an EvalContext to a float (or a dict of floats)
that is logged as a metric. Add a new one here and pass it to
`Evaluator(scalars=[...])`.
"""
from __future__ import annotations

from typing import Literal

import torch

from ..theory import check_condition, condition_mask, measure_order_params
from .evaluator import EvalContext

Positions = Literal["all", "trigg", "non_trigg"]


def _ell_suffix(spec) -> str:
    if spec is None:
        return ""
    if isinstance(spec, str):
        return f"_ell_ge{spec.lstrip('>= ')}"
    return "_ell" + "_".join(str(e) for e in ([spec] if isinstance(spec, int) else spec))


def _mean_or_nan(values: torch.Tensor) -> float:
    """The mean, or NaN when nothing was selected (e.g. an ell no query of the
    test batch has): logged as NaN rather than failing the evaluation."""
    return values.mean().item() if values.numel() else float("nan")


class LossMetric:
    """Mean cross-entropy over a subset of the positions of the test batch.

    positions: "all" (the population loss), "trigg" (trigger queries) or
    "non_trigg" (every other position). ell, only with "trigg": None (every
    trigger query), an int k (ell == k), a list (ell in it) or ">=k" (ell >= k),
    with ell the number of earlier occurrences of the query trigger. The name
    defaults to e.g. "loss", "loss_trigg", "loss_trigg_ell0", "loss_trigg_ell_ge1".
    NaN if no position of the test batch matches.
    """

    def __init__(self, name: str | None = None, positions: Positions = "all", ell=None):
        if positions not in ("all", "trigg", "non_trigg"):
            raise ValueError(f"positions must be 'all', 'trigg' or 'non_trigg', got {positions!r}")
        check_condition(ell)
        if ell is not None and positions != "trigg":
            raise ValueError("ell only applies to positions='trigg'")
        self.positions, self.ell = positions, ell
        default = "loss" if positions == "all" else f"loss_{positions}{_ell_suffix(ell)}"
        self.name = name or default

    def __call__(self, ctx: EvalContext) -> float:
        if self.positions == "all":
            selected = torch.ones_like(ctx.is_trigger)
        elif self.positions == "trigg":
            selected = ctx.is_trigger & condition_mask(ctx.ell, self.ell)
        else:
            selected = ~ctx.is_trigger
        return _mean_or_nan(ctx.token_loss[selected])


class TopKAccuracy:
    """In-context accuracy: the fraction of trigger queries with ell >= 1 (the
    trigger appeared earlier, so the target is in the context) whose target is
    among the top-k logits. NaN if the test batch has no such query."""

    def __init__(self, k: int = 1):
        self.k, self.name = k, f"top{k}_accuracy"

    def __call__(self, ctx: EvalContext) -> float:
        in_context = ctx.trigger_ell >= 1
        top_k = ctx.trigger_logits[in_context].topk(self.k, dim=-1).indices     # (N, k)
        hit = (top_k == ctx.trigger_target[in_context][:, None]).any(dim=-1)
        return _mean_or_nan(hit.float())


class TargetProbMass:
    """Mean softmax probability of the target at the trigger queries with
    ell >= 1. NaN if the test batch has no such query."""

    name = "target_prob"

    def __call__(self, ctx: EvalContext) -> float:
        in_context = ctx.trigger_ell >= 1
        probs = torch.softmax(ctx.trigger_logits[in_context], dim=-1)          # (N, V)
        return _mean_or_nan(probs.gather(1, ctx.trigger_target[in_context][:, None]))


class OrderParameters:
    """Every order parameter registered in `icl.theory.ansatz.ORDER_PARAMS`,
    measured from `ctx.matrices` (the mean of each over its support)."""

    name = "order_parameters"

    def __call__(self, ctx: EvalContext) -> dict[str, float]:
        return measure_order_params(ctx.matrices, ctx.K)
