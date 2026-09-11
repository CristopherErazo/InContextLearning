"""Which training steps get a scalar evaluation and which get artifacts."""
from __future__ import annotations

from typing import Literal

import numpy as np

Scale = Literal["linear", "log"]


def evaluation_steps(total_steps: int, n_points: int, scale: Scale = "linear") -> set[int]:
    """`n_points` distinct integer steps in [0, total_steps], including both ends.

    The training loop evaluates *before* each step, so step 0 is the initial
    model and `total_steps` is the final one (Rewind evaluates it after the
    loop when it is in the schedule; scripts/train.py does the same).
    An `n_points` of 0 disables the schedule.
    """
    if n_points <= 0:
        return set()
    if scale == "linear":
        steps = np.linspace(0, total_steps, num=n_points)
    elif scale == "log":
        # logspace cannot start at 0; prepend it and spread the rest in [1, total_steps]
        steps = np.concatenate([[0], np.logspace(0, np.log10(max(total_steps, 1)), num=max(n_points - 1, 1))])
    else:
        raise ValueError(f"print_scale must be 'linear' or 'log', got {scale!r}")
    return set(np.unique(np.round(steps).astype(int)).tolist())


def get_evaluation_times(args) -> tuple[set[int], set[int]]:
    """(scalar eval steps, artifact steps) from an `ExtraArgs`-like object."""
    return (
        evaluation_steps(args.total_steps, args.n_prints, args.print_scale),
        evaluation_steps(args.total_steps, args.n_prints_model, args.print_scale),
    )
