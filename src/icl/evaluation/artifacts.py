"""Artifact probes: each maps an EvalContext to tensors / objects that are
saved to disk by tracklab. A probe's `group` is the artifact subfolder and
`atype` the tracklab serializer ('tensor' -> .npy, 'pickle' -> .pkl).
Returning a dict saves one artifact per key; returning a single object saves
it under `probe.name`.
"""
from __future__ import annotations

import math

import torch

from .evaluator import EvalContext, split_on_off


class ComposedMatrices:
    """M = Pᵀ WQK1 P, Q = Eᵀ WQK2 WOV1 E, G = U WOV2 E, one .npy each."""

    name, group = "matrices", "matrices"

    def __call__(self, ctx: EvalContext) -> dict[str, torch.Tensor]:
        return ctx.matrices


class PerPositionOnOffLogits:
    """On-target, off-target and all logits at induction-possible positions, at a
    few chosen sequence positions only. Saved as ONE pickle `hists_step_N.pkl` in
    group `logits` holding

        {'fractions': [f, ...], 'positions': [p, ...],
         'on': [array per position], 'off': [...], 'all': [...]}

    where each fraction f of the sequence maps to input position p = ceil(f*L) - 1
    (so f = 1 is the last position) and the i-th array of 'on' / 'off' / 'all'
    holds the sequences whose position `positions[i]` is induction-possible.
    Storing every position cost test_size * L * V floats per save; this is
    len(fractions) / L of that. In pred_mode='last' only position L-1 exists.
    """

    name, group, atype = "hists", "logits", "pickle"

    def __init__(self, fractions=(0.5, 0.75, 1.0)):
        if not fractions or any(not 0 < f <= 1 for f in fractions):
            raise ValueError(f"fractions must be a non-empty list in (0, 1], got {list(fractions)}")
        self.fractions = sorted(set(float(f) for f in fractions))

    def positions(self, L: int) -> list[int]:
        return sorted({math.ceil(f * L) - 1 for f in self.fractions})

    def __call__(self, ctx: EvalContext) -> dict[str, dict[str, list]]:
        _, pos = ctx.ind_index
        L, Lq = ctx.input.size(1), ctx.target.size(1)
        pos = pos + (L - Lq)                  # absolute position ('last' mode keeps only L-1)
        positions = self.positions(L)
        # Select the chosen positions before splitting, so the (N, V-1) off-target
        # copy is only ever made for them.
        keep = torch.isin(pos, torch.tensor(positions, device=pos.device))
        pos, logits = pos[keep], ctx.logits_ind[keep]
        on, off = split_on_off(logits, ctx.target_ind[keep])
        by_pos = lambda t: [t[pos == p].cpu().numpy() for p in positions]
        payload = {"fractions": self.fractions, "positions": positions,
                   "on": by_pos(on), "off": by_pos(off), "all": by_pos(logits)}
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
