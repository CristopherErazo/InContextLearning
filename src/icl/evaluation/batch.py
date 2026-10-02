"""Test-batch preparation for evaluation.

The batch is NOT filtered: every sequence is kept, so the test batch is an
unbiased sample of the task (the loss over all positions estimates the
population loss, and the trigger queries with no earlier occurrence, ell = 0,
keep their true frequency). Everything a probe needs (which positions are
trigger queries, their ell) is derived from the sequences by `EvalContext`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data import occurrence_counts

Batch = dict[str, torch.Tensor | int]


@dataclass(frozen=True)
class PreprocessStats:
    """What the test batch holds, for logging."""
    n_sequences: int          # sequences in the batch
    frac_trigger: float       # fraction of positions holding a trigger query
    frac_in_context: float    # fraction of positions holding a trigger query with ell >= 1

    def summary(self) -> str:
        return (f"test batch: {self.n_sequences} sequences; trigger queries at {self.frac_trigger:.1%} "
                f"of positions, {self.frac_in_context:.1%} with the trigger earlier in context")


def preprocess_batch(batch: Batch, device: str | torch.device = "cpu") -> tuple[Batch, PreprocessStats]:
    """The batch on `device`, and what it holds."""
    batch = {name: (value.to(device) if torch.is_tensor(value) else value) for name, value in batch.items()}
    inputs = batch["sequence"][:, :-1]
    is_trigger = inputs < int(batch["K"])
    in_context = is_trigger & (occurrence_counts(inputs) > 1)
    stats = PreprocessStats(
        n_sequences=int(inputs.shape[0]),
        frac_trigger=is_trigger.float().mean().item(),
        frac_in_context=in_context.float().mean().item(),
    )
    return batch, stats
