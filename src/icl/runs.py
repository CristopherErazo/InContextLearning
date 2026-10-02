"""Reading a finished (or running) run back, for analysis.

    run = RunData("sanity_check", "run_003", base_dir="../data")   # or RunData.last("sanity_check", ...)
    run.config               # TrainerArgs (typed DictConfig), as the run saved it
    run.metrics              # wide DataFrame: one row per step, one column per metric
    run.steps("matrices")    # steps that have artifacts in that group
    run.matrices("last")     # {"M", "Q", "G"} torch tensors (the `model.matrices()` convention)
    run.model(step)          # ReducedTransformer with exactly those weights
    run.batch(4096, seed=0)  # a fresh, unfiltered batch with the run's V, L, K
    run.order_params(step)   # every registered order parameter, measured from the matrices
    run.logit_table(step)    # the TriggerLogitTable probe's QueryTable

`step` is an int, "first" or "last" everywhere. Everything is read lazily and
the tracklab reader stays available as `run.reader` for anything else.
"""
from __future__ import annotations

from functools import cached_property

import pandas as pd
import torch
from omegaconf import OmegaConf
from tracklab import ExperimentReader

from .config import TrainerArgs
from .data import generate_icl_batch
from .reduced import ReducedTransformer
from .theory import QueryTable, measure_order_params


class RunData:
    def __init__(self, experiment: str, run_id: str, base_dir: str = "./data"):
        self.reader = ExperimentReader(experiment, base_dir=base_dir)
        self.run_id = run_id

    @classmethod
    def last(cls, experiment: str, base_dir: str = "./data") -> "RunData":
        """The most recent run of `experiment`."""
        runs = ExperimentReader(experiment, base_dir=base_dir).list_runs()
        if not runs:
            raise FileNotFoundError(f"experiment {experiment!r} in {base_dir!r} has no runs")
        return cls(experiment, runs[-1], base_dir)

    def __repr__(self) -> str:
        return f"RunData({self.reader.experiment_name!r}, {self.run_id!r})"

    # ---- config and metrics --------------------------------------------------

    @cached_property
    def config(self):
        """The saved config as a typed TrainerArgs (derived fields such as K and
        lr are already in it: the run saved them after computing them)."""
        saved = self.reader.load_config(self.run_id)
        return OmegaConf.merge(OmegaConf.structured(TrainerArgs()), saved)

    @cached_property
    def metrics(self) -> pd.DataFrame:
        """One row per evaluated step (the index), one column per metric."""
        long_format = self.reader.load_metrics(self.run_id)
        return long_format.pivot_table(index="step", columns="metric", values="value", aggfunc="last")

    # ---- artifacts -----------------------------------------------------------

    def steps(self, group: str) -> list[int]:
        """Sorted steps at which the run saved artifacts in `group`."""
        index = self.reader.list_artifacts(self.run_id, group)
        return sorted(int(s) for s in index["step"].dropna().unique())

    def load(self, group: str, name: str, step="last"):
        """The artifact `name` of `group` at `step` (int, "first" or "last")."""
        index = self.reader.list_artifacts(self.run_id, group)
        if index.empty:
            raise FileNotFoundError(f"{self}: no artifacts in group {group!r}")
        step = self._resolve(step, group)
        entries = index[(index["name"] == name) & (index["step"] == step)]
        if entries.empty:
            raise FileNotFoundError(f"{self}: no artifact {name!r} in group {group!r} at step {step}")
        return self.reader.load_artifact(self.run_id, entries["file"].iloc[-1], group)

    def _resolve(self, step, group: str) -> int:
        if step in ("first", "last"):
            steps = self.steps(group)
            if not steps:
                raise FileNotFoundError(f"{self}: no artifacts in group {group!r}")
            return steps[0] if step == "first" else steps[-1]
        return int(step)

    def matrices(self, step="last") -> dict[str, torch.Tensor]:
        """{"M", "Q", "G"} saved by the ComposedMatrices probe, as torch tensors."""
        saved = self.load("matrices", "matrices", step)
        return {name: torch.as_tensor(matrix) for name, matrix in saved.items()}

    def model(self, step="last", device=None, dtype=None) -> ReducedTransformer:
        """The model at `step`, rebuilt from its matrices (exact for lin_attn runs,
        of either backend). See `ReducedTransformer.from_matrices`."""
        return ReducedTransformer.from_matrices(self.matrices(step), self.config.model_args,
                                                device=device, dtype=dtype)

    def order_params(self, step="last") -> dict[str, float]:
        """Every order parameter of `icl.theory.ORDER_PARAMS`, measured from the
        matrices at `step` (so also ones added after the run was made)."""
        return measure_order_params(self.matrices(step), self.config.data_args.K)

    def logit_table(self, step="last") -> QueryTable:
        """The logit table saved by the TriggerLogitTable probe at `step`."""
        return QueryTable.from_numpy(self.load("logits", "table", step))

    # ---- data ----------------------------------------------------------------

    def batch(self, num_sequences: int, seed: int | None = None, device="cpu") -> dict:
        """A fresh batch of `num_sequences` sequences from the run's task (V, L, K), with the
        `counts` / `is_trigg` bookkeeping and NOT filtered. `seed` makes it
        reproducible without touching the global RNG."""
        model_args, K = self.config.model_args, self.config.data_args.K
        generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
        return generate_icl_batch(num_sequences, model_args.vocab_size, model_args.seq_len, K,
                                  device=device, generator=generator)
