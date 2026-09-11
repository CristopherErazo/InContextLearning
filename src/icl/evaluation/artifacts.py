"""Artifact probes: each maps an EvalContext to tensors / objects that are
saved to disk by tracklab. A probe's `group` is the artifact subfolder and
`atype` the tracklab serializer ('tensor' -> .npy, 'pickle' -> .pkl).
Returning a dict saves one artifact per key; returning a single object saves
it under `probe.name`.
"""
from __future__ import annotations

import torch

from .evaluator import EvalContext


class ComposedMatrices:
    """M = Pᵀ WQK1 P, Q = Eᵀ WQK2 WOV1 E, G = U WOV2 E, one .npy each."""

    name, group = "matrices", "matrices"

    def __call__(self, ctx: EvalContext) -> dict[str, torch.Tensor]:
        return ctx.matrices


class PerPositionOnOffLogits:
    """On-target, off-target and all logits at induction-possible positions,
    grouped by sequence position. Saved as ONE pickle `hists_step_N.pkl` in
    group `logits` holding {'on': [L arrays], 'off': [...], 'all': [...]},
    the layout the notebooks read back."""

    name, group, atype = "hists", "logits", "pickle"

    def __call__(self, ctx: EvalContext) -> dict[str, dict[str, list]]:
        _, pos = ctx.ind_index
        L = ctx.input.size(1)
        by_pos = lambda t: [t[pos == l].cpu().numpy() for l in range(L)]
        payload = {"on": by_pos(ctx.on_target_logits),
                   "off": by_pos(ctx.off_target_logits),
                   "all": by_pos(ctx.logits_ind)}
        return {self.name: payload}  # one artifact whose data is the whole dict


class LogitHistograms:
    """Normalised histograms of on- and off-target logits on a shared range."""

    name, group = "logit_hist", "logit_hist"

    def __init__(self, n_bins: int = 50):
        self.n_bins = n_bins

    def __call__(self, ctx: EvalContext) -> dict[str, torch.Tensor]:
        on, off = ctx.on_target_logits.cpu(), ctx.off_target_logits.cpu()
        lo, hi = ctx.logits_ind.min().item(), ctx.logits_ind.max().item()
        on_hist, edges = torch.histogram(on, bins=self.n_bins, range=(lo, hi), density=True)
        off_hist, _ = torch.histogram(off.flatten(), bins=self.n_bins, range=(lo, hi), density=True)
        return {"on_hist": on_hist, "off_hist": off_hist, "edges": edges}


class AttentionMaps:
    """Both layers' attention patterns on the test batch."""

    name, group = "attention", "attention"

    def __call__(self, ctx: EvalContext) -> dict[str, torch.Tensor]:
        return {"A1": ctx.attn1, "A2": ctx.attn2}
