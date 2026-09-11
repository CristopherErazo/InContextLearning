# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code studying in-context learning (induction heads) in a minimal two-layer
linear-attention transformer trained on a synthetic trigger-retrieval task. Single
author, exploratory: notebooks are dated experiments, not a product. The repo is
`~/research/projects/in-context-learning/code`; it depends on two sibling repos by the
same author, `~/research/lib/TrackLab` (experiment tracker) and `~/research/lib/Rewind`
(controllable training loop + Shiny dashboard). Both have their own `CLAUDE.md` — read
them when touching tracker / controller / dashboard behaviour.

## Commands

```bash
uv sync                                   # .venv with icl + tracklab + rewind (pinned GitHub) + torch (PyPI)
uv sync --extra cpu|cu126|cu130           # explicit torch build; extras are mutually exclusive

# one training run; any TrainerArgs field overridable with OmegaConf dotted syntax
uv run python -u scripts/launcher.py model_args.vocab_size=512 extra_args.experiment_name=my_exp

# dashboard (launch form + live metrics + pause/resume/set-lr/rewind). Run from repo root:
uv run shiny run --reload scripts/dash.py

# cluster sweep: edit the arrays at the top, then
nohup bash shell/submit.sh > submit.log &
bash shell/monitor.sh 0.5 30              # kills the current launcher once top1_accuracy > threshold

# module self-checks (each has an __main__ block)
uv run python -m icl.data
uv run python -m icl.model

uv run pytest                             # pytest is configured (testpaths=tests) but there is no tests/ yet
```

Run outputs land in `data/<experiment_name>/run_NNN/` (gitignored). `_scratch/` is
gitignored local scratch (old dashboards, one-off scripts); nothing there is imported.

## Dependencies on the sibling libraries

`tracklab` and `rewind` are pinned in `[tool.uv.sources]` to a GitHub tag / commit so a
fresh clone needs no local checkouts (`tool.uv.sources` cannot hold a path fallback: uv
needs the path to exist to validate the lock even for an unselected group, which was
tested and rejected). Consequences:

- **Bumping a lib:** edit the `tag` / `rev` in `pyproject.toml`, then
  `uv lock --upgrade-package <name>` and `uv sync`. Both libs are pinned by tag
  (TrackLab `v1.1.0`, Rewind `v1.0.0`).
- **Editing the libs locally:** after any `uv sync`, run
  `uv pip install -e ../../../lib/TrackLab -e ../../../lib/Rewind`. Then use the
  activated `.venv` or `uv run --no-sync`; a plain `uv run` re-syncs and restores the
  pinned GitHub versions. Check which one is active with `uv pip show tracklab`.
- **Torch:** default is PyPI (CUDA on Linux, CPU on Windows/macOS). The `cpu` / `cu126`
  / `cu130` extras map torch to the PyTorch indexes and are declared as conflicting, so
  the lock covers all of them; never enable two at once. The PyTorch `cu128` index lags
  behind (torch 2.11 while PyPI has 2.14), which is why it is not offered.

## Package layout gotcha

The package body is `src/` **without** an `icl/` subdirectory, but it is installed and
imported as `icl` via `package-dir = {"icl" = "src"}` in `pyproject.toml`. Always
`from icl import ...`, never `from src ...`. When adding a new subpackage under `src/`,
add it to `packages = [...]` in `pyproject.toml` or it will not be installed.
`icl/__init__.py` re-exports everything (config, data, model, utils, and all of
`icl.evaluation.__all__`), which is why `scripts/launcher.py` does `from icl import *`.

## Architecture

**Config (`src/config.py`).** Nested dataclasses `TrainerArgs{model_args, data_args,
optim_args, extra_args}`. The launcher does `OmegaConf.merge(OmegaConf.structured(TrainerArgs()),
OmegaConf.from_cli())` and then **must** call `compute_derived_args` — `data_args.K`
(= `rho * vocab_size`) and `optim_args.lr` (= `alpha_lr * batch_size / d_model`) are
placeholders until then. `ui_metadata` marks fields the Rewind dashboard exposes as
launch-form controls. `extra_args.launch_token` is set only by the dashboard's
`RunLauncher`; the launcher answers it with `rewind.launch.write_handshake` so the
dashboard learns which `run_id` the child claimed.

**Data (`src/data.py`, `generate_icl_batch`).** Trigger tokens are always `0..K-1`
(fixed across the batch); each sequence samples its own K output tokens from `[K, V-1]`
and follows the rule "trigger → its output, anything else → uniform random". Sequences
have length `L+1` (input = `[:, :-1]`, target = `[:, 1:]`). The batch dict carries
`is_trigg` and `counts` (occurrence count of the token at each position), which
downstream code combines into the "induction possible" mask `is_trigg & (counts > 1)`.
The attention `mask` is strictly lower-triangular (`diagonal=-1`): a position never
attends to itself.

**Model (`src/model.py`, `MinimalTransformer`).** Two `FullRankAttentionLayer`s with
no MLP and no residual stream. Layer 1 uses positional embeddings as Q and K and token
embeddings as V (previous-token head); layer 2 uses token embeddings as Q, layer-1
output as K, token embeddings as V (induction head). `lin_attn=True` replaces softmax
with masked scores scaled by `1/sqrt(d*L)`. `initialize_model()` freezes everything
and then unfreezes only `attn1.WQK`, `attn2.WQK`, `attn2.WOV`; E, P, U and `attn1.WOV`
stay at random init. Logits are `beta * U(X2) / sqrt(d)`. `pred_mode="last"` runs
layer 2 only for the final query. `get_composed_matrices()` returns the analysis
objects `M = Pᵀ WQK1 P`, `Q = Eᵀ WQK2 WOV1 E`, `G = U WOV2 E`, keyed as
`(name, "matrices")` tuples — that key shape is what TrackLab's artifact writer expects.

**Evaluation (`src/evaluation/`).** `preprocess_batch` first drops sequences where
induction is never possible (`filter_batch`), then adds `ind_possible`,
`ind_not_possible` and `target_ind_positions` to the batch. `Evaluator` builds one
`EvalContext` (runs the forward pass once under `inference_mode`, exposes logits,
on/off-target logits, and the weight matrices as properties) and calls each metric on
it; a metric returns a scalar (stored under `metric.name`) or a dict (merged). Add new
scalar metrics as callables in `scalar_measures.py`; tensor-valued diagnostics saved as
run artifacts go in `tensor_probes.py`; `theory.py` holds closed-form predictions
(`effective_loss`) compared against runs in notebooks. `get_evaluation_times` turns
`n_prints` / `n_prints_model` / `print_scale` into the eval and artifact step schedules.

**Launcher (`scripts/launcher.py`).** Composition root. `build_controller` wires three
closures — `train_step_fn`, `eval_fn`, `eval_art_fun` — plus a TrackLab run into
`rewind.TrainerController`, then overrides `controller.eval_schedule` /
`eval_artifacts_schedule` with the computed step sets. `enable_control` attaches a
`RunMailbox` (dashboard pause/resume/rewind commands), `enable_rewind` turns on
snapshotting, `track_artifacts` enables the artifact closure. `scripts/dash.py` is a
thin `rewind.dashboard.build_dashboard(DashboardConfig(...))` that spawns new runs as
`python -m scripts.launcher`, so it must be launched from the repo root.

**Notebooks.** Named `YY_MM_DD_Topic.ipynb`; they read runs back with
`tracklab.ExperimentReader(experiment_name, base_dir='../data')`. Several still import
a `configurations` plotting package that was removed from this repo (the author intends
to move it into `~/research/lib/`); expect those cells to fail until that lands.
