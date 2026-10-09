# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code studying in-context learning (induction heads) in a minimal two-layer
linear-attention transformer trained on a synthetic trigger-retrieval task. Single
author, exploratory. It depends on two libraries by the same author, TrackLab (experiment
tracker) and Rewind (controllable training loop + Shiny dashboard), pinned to GitHub in
`pyproject.toml`; both have their own `CLAUDE.md` — read them when touching tracker /
controller / dashboard behaviour. Setup is in `README.md`. Project status and decisions
live in the project root's `journal/`, the experiment log in `journal/RUNS.md`.

## Commands

```bash
uv sync                                   # .venv with icl + tracklab + rewind (pinned GitHub) + torch (PyPI)
uv sync --extra cpu|cu126|cu130           # explicit torch build; extras are mutually exclusive

# one training run; any TrainerArgs field overridable with OmegaConf dotted syntax
uv run python -u scripts/launcher.py model_args.vocab_size=512 extra_args.experiment_name=my_exp
# same run, plain loop with TrackLab only (no Rewind: no dashboard / pause / rewind)
uv run python -u scripts/train.py model_args.vocab_size=512 extra_args.experiment_name=my_exp
# same model in the reduced (M, Q, G) coordinates: exact for SGD, cost independent of d
uv run python -u scripts/train.py model_args.backend=reduced model_args.init=full ...
uv run python -u scripts/train.py model_args.backend=reduced model_args.init=sample model_args.infinite_d=true ...

# dashboard (launch form + live metrics + pause/resume/set-lr/rewind). Run from repo root:
uv run shiny run --reload scripts/dash.py

# cluster sweep: edit the arrays at the top, then
nohup bash shell/submit.sh > submit.log &

# module self-checks (each has an __main__ block)
uv run python -m icl.data
uv run python -m icl.model

uv run pytest                             # testpaths=tests
```

Run outputs land in `data/<experiment_name>/run_NNN/` (gitignored). `_scratch/` is
gitignored local scratch; nothing there is imported.

**Dependencies.** `tracklab` / `rewind` are pinned by tag in `[tool.uv.sources]` (no path
fallback: uv needs the path to exist even for an unselected group). To bump one: edit the
tag, `uv lock --upgrade-package <name>`, `uv sync`. To work against local checkouts:
`uv pip install -e <path>/TrackLab -e <path>/Rewind` after every `uv sync`, then use the
activated `.venv` or `uv run --no-sync` — a plain `uv run` re-syncs to the pins (check with
`uv pip show tracklab`). Never enable two torch extras at once; `cu128` is not offered
because that index lags behind PyPI.

## Package layout

Src layout: the package is `src/icl/`, installed editable by `uv sync`; always
`from icl import ...`. A new subpackage only needs an `__init__.py`. `icl/__init__.py`
re-exports everything (config, data, model, utils, training, `icl.evaluation.__all__`,
`icl.theory`); scripts import their names explicitly from `icl`.

## Architecture

Module docstrings hold the details; below are the invariants and the traps.

**Config (`config.py`).** Nested dataclasses `TrainerArgs{model_args, data_args, optim_args,
extra_args, flow_args}`. `load_config()` merges the structured defaults with
`OmegaConf.from_cli()` and calls `compute_derived_args`: `data_args.K` (= `rho * vocab_size`)
and `optim_args.lr` (= `alpha_lr * batch_size / d_model`) are placeholders until then, so
anything that builds a config by hand must call it too. `ui_metadata` marks the fields the
Rewind dashboard exposes. `flow_args` (`FlowArgs`) is empty for training runs; it exists so
effective flows saved as TrackLab runs read back through `RunData` — a stopgap until `RunData`
reads non-training runs: don't build more on it, propose the generalisation when touching `RunData`.

**Data (`data.py`, `generate_icl_batch`).** Triggers are always tokens `0..K-1`; each
sequence draws its K outputs from `[K, V-1]`; rule "trigger → its output, else uniform".
Sequences have length `L+1` (input `[:, :-1]`, target `[:, 1:]`). The trigger set is never
materialised: "token `< K`" is the trigger test everywhere (evaluation and `icl.theory`
rely on it). The loop-free generator relies on outputs never being triggers (no two
adjacent triggers, closed-form chain): if outputs ever overlap the trigger set, it is
wrong. Position 0 is drawn from the stationary law. `stats=False` (training) skips
`is_trigg` / `counts`; evaluation recomputes them from `sequence` (ell = counts - 1).
`data_args.gen_device` defaults to the training device (2–35x faster on CUDA on the A100).

**Mask.** Not part of a batch: build it with `icl.make_mask(kind, seq_len, device)`,
strictly lower-triangular (`"causal"`: `j < i`; `"prev"`: `j == i-1`). Code reading
`batch['mask']` predates this and fails.

**Model (`model.py`, `MinimalTransformer`).** Two attention layers, no MLP, no residual.
Layer 1: positions as Q, K, tokens as V (previous-token head); layer 2: tokens as Q,
layer-1 output as K, tokens as V (induction head). Each layer owns its mask
(`model_args.mask1` / `mask2`; `mask1=prev` is the intended ablation). `lin_attn=True`
scales masked scores by `1/sqrt(d*L)`. Only `attn1.WQK`, `attn2.WQK`, `attn2.WOV` train.
`matrices()` returns the plain dict `{"M", "Q", "G"}` (CPU, normalised by √d as in the
paper; logits `= beta/L Σ M Q G`): the one matrix convention of the package (probes,
artifacts, `RunData`, `ReducedTransformer.from_matrices`).

**Reduced backend (`reduced.py`, `model_args.backend="reduced"`).** The same SGD
dynamics in (M, Q, G) coordinates with fixed Gram matrices; stores the normalised m, q, g,
so a step costs O(B L² V) for any d and `d = inf` is `k = I`. SGD only (refuses Adam,
`lin_attn=False`, dropout). `init="full"` reproduces a full run's draw; `init="sample"`
draws Grams and m, q, g directly (`infinite_d=true`); `ansatz_init=true` projects the draw
on the ansatz (`ansatz_keep_spread` keeps named matrices; `build_model` needs `K=`).
`from_matrices(mats, model_args)` rebuilds the exact forward pass of any lin_attn run from
saved matrices. Only `scripts/train.py` builds it; `launcher.py` refuses it.
`tests/test_reduced.py` checks full vs reduced trajectories to 1e-9 (float64).

**Reading runs (`runs.py`, `RunData`).** The analysis entry point; API in the module
docstring. The run id is a string: `RunData("full_ansatz", "run_001", base_dir="data")`.
`step` is an int, `"first"` or `"last"`.

**Effective model (`src/icl/theory/`).** The ansatz, its logit moments, the effective loss
and its flow. Its own `CLAUDE.md` (`src/icl/theory/CLAUDE.md`) loads when you work there.

**Training helpers (`training.py`).** `build_model(model_args, optim_args, device)` →
`(model, optimizer, log_line)` for either backend; `get_optimizer`, `compute_loss`.
`track_results` records `train_time`, `eval_time`, `ms_per_step`, `peak_gpu_mem_gib`.

**Evaluation (`src/icl/evaluation/`).** One module per job:
- `batch.py`: `preprocess_batch` moves the batch, does NOT filter it (`loss` estimates the
  population loss); probes select positions by `is_trigger` and `ell`.
- `evaluator.py`: `EvalContext(model, batch, step, chunk)` is a lazy view of the model on
  the test batch: one chunked forward pass under `inference_mode` keeps only `token_loss`
  (B, L) and `trigger_logits` (N, V); `matrices`, `trigger_*` are computed at most once, on
  demand; `outputs` / `attn1/2` are one unchunked `full_output` call. A *probe* is any
  callable with a `name` taking an `EvalContext`. `Evaluator(scalars, artifacts, chunk)`
  memoizes the context on `step`, so scalars and artifacts at the same step share one
  forward pass; artifact probes run when their class attribute `schedule` matches
  (`"artifact"` default, `"scalar"` = at every scalar evaluation). `log_artifacts` writes
  to a TrackLab run (Rewind consumes the same dict).
- `scalars.py`: scalar probes — add new metrics here. `LossMetric(positions, ell)`
  (`ell` only with `"trigg"`; NaN when nothing matches), `TopKAccuracy`, `TargetProbMass`
  (trigger queries with ell >= 1), `OrderParameters`, `BlockVariances`;
  `order_parameter_probes(extra_args.log_order_params)` builds the order-parameter probes.
- `artifacts.py`: artifact probes (class attributes `group`, `atype`, `schedule`; a dict
  result saves one artifact per key): `ComposedMatrices` (`matrices/`), `TriggerLogitTable`
  (`logits/`, at `extra_args.logit_positions`), `MProfile` (`profile/`, schedule `"scalar"`).
- `schedule.py`: `get_evaluation_times` → scalar and artifact step sets over
  `[0, total_steps]`; evaluation at step s sees the weights before the s-th update.

**Launcher (`scripts/launcher.py`).** Composition root for controllable runs: wires
`train_step_fn`, `eval_fn`, `eval_artifacts_fn` and a TrackLab run into
`rewind.TrainerController` and sets its eval schedules; the two eval closures share one
`EvalContext` per step. Terminal logging only when stdout is a TTY. `scripts/dash.py`
spawns runs as `python -m scripts.launcher`, so launch it from the repo root.

**Plain trainer (`scripts/train.py`).** Same model, probes, schedule and artifact layout,
TrackLab only, no Rewind; the only script that runs `backend=reduced`. When changing what a
run evaluates or saves, change both scripts. `extra_args.eval_chunk` (default: the training
batch size) goes to the `Evaluator`; `extra_args.stop_at_loss` / `stop_at_accuracy` stop
early. Other scripts in `scripts/` and `shell/` are one-off experiments, logged in the
project root's `journal/RUNS.md`.

## Branches and notebooks

`new-ansatz` is the active development branch (from `main`). `main` is kept lean (`src/`,
`tests/`, `scripts/{train,launcher,dash}.py`, core `shell/` scripts, `notebooks/template.ipynb`);
`code-optimization` archives old dated notebooks, benchmarks and shell helpers.

Notebooks in `notebooks/` belong to the user: read only unless asked. Dated ones are named
`YY-MM-DD_Topic.ipynb` and read runs with `RunData` or
`tracklab.ExperimentReader(experiment_name, base_dir='../data')`.
