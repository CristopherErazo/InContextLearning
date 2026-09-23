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


class MOrderParameters:
    """Order parameters of `M = P^T WQK1 P`, whose active part is the strictly
    lower triangle (a query attends only to j < i): `M_on` is the first
    sub-diagonal, `M_off` everything below it."""

    name = "M_order_parameters"
    names = ("M_on", "M_off")

    def __call__(self, ctx: EvalContext) -> dict[str, float]:
        M = ctx.matrices["M"]                                    # (L, L)
        L = M.size(0)
        scale = ctx.model.embed.d_model ** 0.5
        on = M.diagonal(-1).sum() / (scale * (L - 1))            # M_{mu, mu-1}, mu = 2..L
        off = 2 * M.tril(-2).sum() / (scale * (L - 1) * (L - 2))  # mu = 3..L, nu <= mu-2
        return {"M_on": on.item(), "M_off": off.item()}


class QOrderParameters:
    """Order parameters of `Q = E^T WQK2 WOV1 E`: the trigger diagonal, the
    off-diagonal trigger rows, and the non-trigger rows."""

    name = "Q_order_parameters"
    names = ("Q_on", "Q_T", "Q_noT")

    def __call__(self, ctx: EvalContext) -> dict[str, float]:
        Q = ctx.matrices["Q"]                                    # (V, V)
        V = Q.size(0)
        scale = ctx.model.embed.d_model ** 0.5
        trig, K = ctx.trigger_mask, ctx.K
        diag = Q.diagonal()
        on = diag[trig].sum() / (scale * K)
        q_t = (Q[trig].sum() - diag[trig].sum()) / (scale * K * (V - 1))
        q_no = Q[~trig].sum() / (scale * V * (V - K))
        return {"Q_on": on.item(), "Q_T": q_t.item(), "Q_noT": q_no.item()}


class GammaOrderParameters:
    """Order parameters of `Gamma = U WOV2 E`: the non-trigger diagonal, the
    trigger rows, and the off-diagonal non-trigger rows."""

    name = "G_order_parameters"
    names = ("G_on", "G_T", "G_noT")

    def __call__(self, ctx: EvalContext) -> dict[str, float]:
        G = ctx.matrices["G"]                                    # (V, V)
        V = G.size(0)
        scale = ctx.model.embed.d_model ** 0.5
        trig, K = ctx.trigger_mask, ctx.K
        diag = G.diagonal()
        on = diag[~trig].sum() / (scale * (V - K))
        g_t = G[trig].sum() / (scale * K * V)
        g_no = (G[~trig].sum() - diag[~trig].sum()) / (scale * (V - K) * (V - 1))
        return {"G_on": on.item(), "G_T": g_t.item(), "G_noT": g_no.item()}
