"""Scalar probes: each maps an EvalContext to a float (or a dict of floats)
that is logged as a metric. Add a new one here and pass it to
`Evaluator(scalars=[...])`.
"""
from __future__ import annotations

from typing import Literal

import torch

from ..theory import (DIAGNOSTIC_MEANS, ORDER_PARAMS, POOLED_VARIANCES, VARIANCES, check_condition, condition_mask,
                      measure_diagnostic_means, measure_order_params, measure_variances)
from .artifacts import MProfile
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


MEAN_NAMES = (*ORDER_PARAMS, *DIAGNOSTIC_MEANS)
VARIANCE_NAMES = (*VARIANCES, *POOLED_VARIANCES, "var_M_on_raw")


def _checked(names, allowed) -> list[str]:
    names = list(allowed if names is None else dict.fromkeys(names))
    unknown = set(names) - set(allowed)
    if unknown:
        raise ValueError(f"unknown names {sorted(unknown)}; allowed: {list(allowed)}")
    return names


class OrderParameters:
    """The block means measured from `ctx.matrices` (the mean of each over its
    support): by default every order parameter of `icl.theory.ORDER_PARAMS`;
    `names` may also pick the diagnostic means of `DIAGNOSTIC_MEANS` (Q_N, G_N,
    zero in the ansatz)."""

    name = "order_parameters"

    def __init__(self, names=None):
        self.names = _checked(list(ORDER_PARAMS) if names is None else names, MEAN_NAMES)

    def __call__(self, ctx: EvalContext) -> dict[str, float]:
        means = measure_order_params(ctx.matrices, ctx.K)
        if any(name in DIAGNOSTIC_MEANS for name in self.names):
            means.update(measure_diagnostic_means(ctx.matrices, ctx.K))
        return {name: means[name] for name in self.names}


class BlockVariances:
    """The block variances of `icl.theory.measure_variances`, measured from
    `ctx.matrices`: the eight of VARIANCES, the pooled ones of POOLED_VARIANCES and
    "var_M_on_raw" (all by default, or only `names`)."""

    name = "block_variances"

    def __init__(self, names=None):
        self.names = _checked(names, VARIANCE_NAMES)

    def __call__(self, ctx: EvalContext) -> dict[str, float]:
        variances = measure_variances(ctx.matrices, ctx.K)
        return {name: variances[name] for name in self.names}


ORDER_PARAMETER_GROUPS = {"means": MEAN_NAMES, "variances": VARIANCE_NAMES, "profile": ("profile",)}


def order_parameter_probes(names) -> tuple[list, list]:
    """(scalar probes, artifact probes) for a list such as `extra_args.log_order_params`:
    the groups "means" (OrderParameters of the six order parameters and Q_N, G_N),
    "variances" (BlockVariances of every block variance) and "profile" (MProfile, the
    sub-diagonal of M, saved at every scalar evaluation), and/or individual metric
    names. An empty list gives no probe."""
    chosen = []
    for name in names:
        if name not in ORDER_PARAMETER_GROUPS and name not in (*MEAN_NAMES, *VARIANCE_NAMES):
            raise ValueError(f"unknown order-parameter log {name!r}: use a group of "
                             f"{list(ORDER_PARAMETER_GROUPS)} or a name of {[*MEAN_NAMES, *VARIANCE_NAMES]}")
        chosen.extend(ORDER_PARAMETER_GROUPS.get(name, (name,)))
    means = [name for name in dict.fromkeys(chosen) if name in MEAN_NAMES]
    variances = [name for name in dict.fromkeys(chosen) if name in VARIANCE_NAMES]
    scalars = ([OrderParameters(means)] if means else []) + ([BlockVariances(variances)] if variances else [])
    return scalars, ([MProfile()] if "profile" in chosen else [])
