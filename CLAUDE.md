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
# same run, plain loop with TrackLab only (no Rewind: no dashboard / pause / rewind)
uv run python -u scripts/train.py model_args.vocab_size=512 extra_args.experiment_name=my_exp

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

## Package layout

Standard src layout: the package body is `src/icl/` (`package-dir = {"" = "src"}` in
`pyproject.toml`), installed editable by `uv sync`, so `import icl` only ever resolves
to the installed package and never to a folder in the working directory. Always
`from icl import ...`. When adding a new subpackage under `src/icl/`, add it to
`packages = [...]` in `pyproject.toml` or it will not be installed. `icl/__init__.py`
re-exports everything (config, data, model, utils, training, and all of
`icl.evaluation.__all__`); scripts import their names explicitly from `icl`.
`.vscode/settings.json` points the editor at `.venv` so Pylance resolves `icl`,
`tracklab` and `rewind`.

## Architecture

**Config (`src/icl/config.py`).** Nested dataclasses `TrainerArgs{model_args, data_args,
optim_args, extra_args}`. Both scripts start from `load_config()`, which merges
`OmegaConf.structured(TrainerArgs())` with `OmegaConf.from_cli()` and then calls
`compute_derived_args` — `data_args.K` (= `rho * vocab_size`) and `optim_args.lr`
(= `alpha_lr * batch_size / d_model`) are placeholders until then; anything that builds
a config by hand must call it too. `ui_metadata` marks fields the Rewind dashboard exposes as
launch-form controls. `extra_args.launch_token` is set only by the dashboard's
`RunLauncher`; the launcher answers it with `rewind.launch.write_handshake` so the
dashboard learns which `run_id` the child claimed.

**Data (`src/icl/data.py`, `generate_icl_batch`).** Trigger tokens are always `0..K-1`
(fixed across the batch); each sequence samples its own K output tokens from `[K, V-1]`
and follows the rule "trigger → its output, anything else → uniform random". Sequences
have length `L+1` (input = `[:, :-1]`, target = `[:, 1:]`). The batch dict carries
`is_trigg` and `counts` (occurrence count of the token at each position), which
downstream code combines into the "induction possible" mask `is_trigg & (counts > 1)`.
The attention `mask` is strictly lower-triangular (`diagonal=-1`): a position never
attends to itself.

**Model (`src/icl/model.py`, `MinimalTransformer`).** Two `FullRankAttentionLayer`s with
no MLP and no residual stream. Layer 1 uses positional embeddings as Q and K and token
embeddings as V (previous-token head); layer 2 uses token embeddings as Q, layer-1
output as K, token embeddings as V (induction head). `lin_attn=True` replaces softmax
with masked scores scaled by `1/sqrt(d*L)`. `initialize_model()` freezes everything
and then unfreezes only `attn1.WQK`, `attn2.WQK`, `attn2.WOV`; E, P, U and `attn1.WOV`
stay at random init. Logits are `beta * U(X2) / sqrt(d)`. `pred_mode="last"` runs
layer 2 only for the final query. `get_composed_matrices()` returns the analysis
objects `M = Pᵀ WQK1 P`, `Q = Eᵀ WQK2 WOV1 E`, `G = U WOV2 E`, keyed as
`(name, "matrices")` tuples — that key shape is what TrackLab's artifact writer expects.

**Training helpers (`src/icl/training.py`).** `get_optimizer(params, optim_args)` and
`compute_loss(model, batch, loss_fn, device)` (the training forward pass, honouring
`pred_mode`). Shared by both scripts; not part of `icl.evaluation`.

**Evaluation (`src/icl/evaluation/`).** One module per job:
- `batch.py`: `preprocess_batch(batch, device)` drops sequences where induction is never
  possible (`filter_batch`) and adds the `ind_possible` mask (`is_trigg & counts > 1`).
  It returns `(batch, PreprocessStats)`; nothing else is precomputed.
- `evaluator.py`: the probing machinery. `EvalContext(model, batch, loss_fn, step)` is a
  lazy view of the model on the test batch: `outputs` (one `model.full_output` call under
  `inference_mode`, train mode restored), `logits`, `attn1/2`, `ind_index`, `logits_ind`,
  `target_ind`, `on/off_target_logits`, `matrices` (from `model.get_composed_matrices`)
  are `cached_property`s, so each is computed at most once and only if a probe asks. A
  *probe* is any callable with a `name` taking an `EvalContext` (the `Probe` protocol).
  `Evaluator(scalars=[...], artifacts=[...], loss_fn)` memoizes the context on `step`:
  `scalars(model, batch, step)` returns `dict[str, float]` (a probe may return a dict,
  merged), `artifacts(model, batch, step)` returns `{(name, group): (data, type)}`, and
  both calls at the same step share one forward pass. Tensors are moved to CPU / numpy
  during packing, so probes stay device-agnostic. `log_artifacts(run, artifacts, step)`
  writes that dict to a TrackLab run (Rewind consumes the same dict directly).
- `scalars.py`: scalar probes (`LossMetric`, `TopKAccuracy`, `TargetProbMass`,
  `LogitStatistics`). Add new metrics here.
- `artifacts.py`: artifact probes; class attributes `group` (artifact subfolder) and
  `atype` (TrackLab serializer, default `tensor`). A dict result saves one artifact per
  key, so a probe that wants a single pickled dict returns `{self.name: payload}`
  (`PerPositionOnOffLogits` does this to keep the `logits/hists_step_N.pkl` layout the
  notebooks read; `ComposedMatrices` writes `matrices/{M,Q,G}_step_N.npy`).
- `schedule.py`: `get_evaluation_times(extra_args)` turns `n_prints` / `n_prints_model`
  / `print_scale` into two step sets spanning `[0, total_steps]` inclusive. Evaluation
  at step `s` sees the weights before the s-th update, so `total_steps` is the final
  model; both loops evaluate it after the last update when it is in the schedule.

**Launcher (`scripts/launcher.py`).** Composition root for controllable runs.
`build_controller` wires three closures — `train_step_fn`, `eval_fn`, `eval_artifacts_fn`
— plus a TrackLab run into `rewind.TrainerController`, then sets
`controller.eval_schedule` / `eval_artifacts_schedule` to the computed step sets. The
two eval closures call `evaluator.scalars/artifacts(..., step=controller.step)`: Rewind
runs them back to back at the same step, so they share one `EvalContext`. `enable_control`
attaches a `RunMailbox` (dashboard pause/resume/rewind commands), `enable_rewind` turns on
snapshotting, `track_artifacts` enables the artifact closure. Terminal logging is on only
when stdout is a TTY (dashboard-spawned and `nohup` runs log to file only). `scripts/dash.py`
is a thin `rewind.dashboard.build_dashboard(DashboardConfig(...))` that spawns new runs as
`python -m scripts.launcher`, so it must be launched from the repo root.

**Plain trainer (`scripts/train.py`).** The same model, probes, schedule and artifact
layout with the loop written out and only TrackLab used; no Rewind import. Use it when the
dashboard / rewind machinery is not needed or to check a result independently of Rewind.
When changing what a run evaluates or saves, change both scripts.

**Notebooks.** Named `YY_MM_DD_Topic.ipynb`; they read runs back with
`tracklab.ExperimentReader(experiment_name, base_dir='../data')`. Several still import
a `configurations` plotting package that was removed from this repo (the author intends
to move it into `~/research/lib/`); expect those cells to fail until that lands.
