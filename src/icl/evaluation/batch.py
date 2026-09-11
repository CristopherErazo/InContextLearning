"""Test-batch preparation for evaluation.

`generate_icl_batch` gives raw sequences plus the `is_trigg` / `counts`
bookkeeping. Evaluation additionally needs to know at which positions the
model *could* retrieve the answer from context ("induction possible": the
input token is a trigger that has already appeared earlier in the sequence).
`preprocess_batch` drops sequences where that never happens and attaches the
`ind_possible` mask that `EvalContext` builds everything else from.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

Batch = dict[str, torch.Tensor]


@dataclass(frozen=True)
class PreprocessStats:
    """What `preprocess_batch` did to the batch, for logging."""
    n_sequences: int          # sequences kept after filtering
    frac_kept: float          # kept / generated
    frac_ind_possible: float  # fraction of (sequence, position) pairs where induction is possible

    def summary(self) -> str:
        return (f"test batch: {self.n_sequences} sequences kept ({self.frac_kept:.1%}); "
                f"induction possible at {self.frac_ind_possible:.1%} of positions")


def induction_mask(batch: Batch) -> torch.Tensor:
    """(B, L) bool: input token is a trigger that already occurred earlier."""
    return batch["is_trigg"].bool() & (batch["counts"] > 1)


def filter_batch(batch: Batch, device: str | torch.device = "cpu") -> tuple[Batch, float]:
    """Keep only sequences with at least one induction-possible position.

    Returns the filtered batch (on `device`) and the fraction of sequences kept.
    """
    keep = induction_mask(batch).any(dim=-1)  # (B,)
    filtered = {k: v.to(device)[keep] for k, v in batch.items()}
    return filtered, keep.float().mean().item()


def preprocess_batch(batch: Batch, device: str | torch.device = "cpu") -> tuple[Batch, PreprocessStats]:
    """Filter the batch and add the `ind_possible` mask.

    Everything downstream (targets at induction positions, on/off-target logits,
    per-position statistics) is derived lazily by `EvalContext` from this mask,
    so nothing else needs to be precomputed here.
    """
    batch, frac_kept = filter_batch(batch, device)
    ind_possible = induction_mask(batch)  # (B, L)
    batch["ind_possible"] = ind_possible
    stats = PreprocessStats(
        n_sequences=int(ind_possible.shape[0]),
        frac_kept=frac_kept,
        frac_ind_possible=ind_possible.float().mean().item(),
    )
    return batch, stats
