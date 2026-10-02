"""Artifact probes: each maps an EvalContext to tensors / objects that are
saved to disk by tracklab. A probe's `group` is the artifact subfolder and
`atype` the tracklab serializer ('tensor' -> .npy, 'pickle' -> .pkl).
Returning a dict saves one artifact per key; returning a single object saves
it under `probe.name`.
"""
from __future__ import annotations

import math

import torch

from ..theory import logit_table
from .evaluator import EvalContext


class ComposedMatrices:
    """`model.matrices()` saved as ONE pickle `matrices/matrices_step_N.pkl` holding
    the plain dict {"M", "Q", "G"} of numpy arrays (convention in
    `MinimalTransformer.matrices`). Read it back with `icl.RunData.matrices`, or
    with tracklab alone: `reader.load_artifact(run_id, "matrices_step_N.pkl", "matrices")`.
    """

    name, group, atype = "matrices", "matrices", "pickle"

    def __call__(self, ctx: EvalContext) -> dict[str, dict]:
        return {self.name: {name: matrix.numpy() for name, matrix in ctx.matrices.items()}}


class TriggerLogitTable:
    """`icl.theory.logit_table` of the model on the test batch, at the positions
    mu = ceil(f * L) for f in `fractions`: ONE pickle `logits/table_step_N.pkl`
    holding the QueryTable as a plain dict of numpy arrays ("logits"
    (num_rows, V) in the canonical layout, "mu", "ell", "sequence_index", "K").
    Read it back with `icl.RunData.logit_table`. Every trigger query is kept
    (ell = 0 included); for more queries than the test batch holds, use
    `logit_table` on a fresh batch (`RunData.batch`).
    """

    name, group, atype = "table", "logits", "pickle"

    def __init__(self, fractions=(0.5, 0.75, 1.0)):
        if not fractions or any(not 0 < f <= 1 for f in fractions):
            raise ValueError(f"fractions must be a non-empty list in (0, 1], got {list(fractions)}")
        self.fractions = sorted(set(float(f) for f in fractions))

    def mus(self, L: int) -> list[int]:
        return sorted({math.ceil(f * L) for f in self.fractions})

    def __call__(self, ctx: EvalContext) -> dict[str, dict]:
        table = logit_table(ctx.model, ctx.batch, self.mus(ctx.input.size(1)), chunk=ctx.chunk)
        return {self.name: table.to_numpy()}


class AttentionMaps:
    """Both layers' attention patterns on the test batch."""

    name, group = "attention", "attention"

    def __call__(self, ctx: EvalContext) -> dict[str, torch.Tensor]:
        return {"A1": ctx.attn1, "A2": ctx.attn2}
