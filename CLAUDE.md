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
# same model in the reduced (M, Q, G) coordinates: exact for SGD, cost independent of d
uv run python -u scripts/train.py model_args.backend=reduced model_args.init=full ...
uv run python -u scripts/train.py model_args.backend=reduced model_args.init=sample model_args.infinite_d=true ...

# dashboard (launch form + live metrics + pause/resume/set-lr/rewind). Run from repo root:
uv run shiny run --reload scripts/dash.py

# cluster sweep: edit the arrays at the top, then
nohup bash shell/submit.sh > submit.log &
bash shell/monitor.sh 0.5 30              # kills the current launcher once top1_accuracy > threshold

# module self-checks (each has an __main__ block)
uv run python -m icl.data
uv run python -m icl.model
uv run python -u scripts/bench_data.py --step   # loop vs vectorised sampler, CPU vs CUDA

uv run pytest                             # tests/ holds the data-generator checks (testpaths=tests)
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

Standard src layout: the package body is `src/icl/`, discovered automatically by
`[tool.setuptools.packages.find] where = ["src"]` in `pyproject.toml` and installed
editable by `uv sync`, so `import icl` only ever resolves to the installed package and
never to a folder in the working directory. Always `from icl import ...`. A new
subpackage under `src/icl/` only needs an `__init__.py` to be installed. `icl/__init__.py`
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
have length `L+1` (input = `[:, :-1]`, target = `[:, 1:]`). The trigger set is never
materialised: the batch carries only the scalar `K`, and "token `< K`" is the
trigger test everywhere (evaluation depends on this). The batch dict also carries
`is_trigg` and `counts` (occurrence count of the token at each position), which
downstream code combines into the "induction possible" mask `is_trigg & (counts > 1)`.

The generator is loop-free. Because outputs are drawn from `[K, V-1]` they are never
triggers, so no two triggers can be adjacent, and the chain has the closed form
`T_t = b_t ∧ ¬T_{t-1}`, `x_t = output_set[u_{t-1}] if T_{t-1} else u_t` with
`u_t` i.i.d. uniform and `b_t = 1{u_t < K}`; the trigger flags follow from run parity
(a `cummax`) and `counts` from a stable argsort. That invariant is load-bearing — if
outputs ever overlap the trigger set, this derivation is void. Position 0 is drawn from
the chain's stationary law, which is exactly weight `2/(V+K)` on output tokens and
`1/(V+K)` elsewhere. `stats=False` returns only `sequence` / `K` / `output_set`
and skips the `counts` work; training passes it, evaluation does not.
`device=` draws the batch there directly and `data_args.gen_device` (`"cpu"`, `"cuda"`
or `"auto"` = the training device, the default) is what the scripts pass — measure
with `scripts/bench_data.py` before changing it. On the A100 workstation the
vectorised generator is 2–35x faster on CUDA than on CPU and never loses a train
step, which is why the default is no longer `"cpu"`. Passing a `torch.Generator` gives a
reproducible stream independent of the global seed.

**The attention mask is not part of a batch.** It is strictly lower-triangular
(`diagonal=-1`: a position never attends to itself) and constant, so carrying it per
batch cost `B*L*L` bytes — 78–95% of every batch — for no information. Build one with
`icl.make_mask(kind, seq_len, device)`: `"causal"` keeps `j < i`, `"prev"` keeps only
`j == i-1`. Notebooks that read `batch['mask']` predate this and will fail at that line.

**Model (`src/icl/model.py`, `MinimalTransformer`).** Two `FullRankAttentionLayer`s with
no MLP and no residual stream. Layer 1 uses positional embeddings as Q and K and token
embeddings as V (previous-token head); layer 2 uses token embeddings as Q, layer-1
output as K, token embeddings as V (induction head). Each layer owns its mask as a
non-persistent `(L, L)` bool buffer built from `model_args.mask1` / `mask2`, so the two
can differ — `mask1=prev` restricts layer 1 to the previous token alone while layer 2
stays `causal`, which is the intended ablation. `forward` / `full_output` still accept
an optional `mask=` that overrides both. `lin_attn=True` replaces softmax
with masked scores scaled by `1/sqrt(d*L)`; the softmax path zeroes rows that have
nothing to attend to (position 0 under either mask) instead of returning NaN, which is
what the linear path already did. `initialize_model()` freezes everything
and then unfreezes only `attn1.WQK`, `attn2.WQK`, `attn2.WOV`; E, P, U and `attn1.WOV`
stay at random init. Logits are `beta * U(X2) / sqrt(d)`. `pred_mode="last"` runs
layer 2 only for the final query. `get_composed_matrices()` returns the analysis
objects `M = Pᵀ WQK1 P`, `Q = Eᵀ WQK2 WOV1 E`, `G = U WOV2 E`, keyed as
`(name, "matrices")` tuples — that key shape is what TrackLab's artifact writer expects.

**Reduced backend (`src/icl/reduced.py`, `model_args.backend="reduced"`).** The same
model trained by SGD, rewritten in the coordinates the loss depends on. The logits
depend on the trained weights only through M (L×L), Q, G (V×V), and an SGD step on
(WQK1, WQK2, WOV2) moves them exactly by `M -= eta K_P dM K_P`, `Q -= eta K_E dQ K_R`,
`G -= eta K_U dG K_E`, with fixed Gram matrices `K_P = PPᵀ`, `K_E = EEᵀ`, `K_U = UUᵀ`,
`K_R = E WOV1ᵀ WOV1 Eᵀ`. `ReducedTransformer` stores everything normalised
(`m = M/√d`, `k = K/d`), which leaves no d in the forward pass: a step costs
O(B L² V) whatever d_model is, and `d = inf` is `k = I`. `ReducedSGD` applies the
Gram-scaled step with `lr = eta_0 = alpha_lr` (momentum and weight decay map across
exactly; Adam does not, so the backend refuses anything but SGD, and also
`lin_attn=False` and dropout). `model_args.init="full"` composes an initialised
`MinimalTransformer` (the same random draw a full run makes, so old runs can be
reproduced; the d×d weights have to fit once), and `"sample"` draws the Gram matrices
(Bartlett) and initial m, q, g from their joint law directly (any d ≥ max(L, V), and
`model_args.infinite_d=true`). `tests/test_reduced.py` checks the full and reduced SGD
trajectories agree to 1e-9 in float64 and the sampled init has the full init's
moments. Rerunning `large_d_sweep_1/run_005` (d=2048, L=512) with `init=full` gave
the same early-stop step with metrics equal to ~1e-4 relative (TF32), at 25 vs 356
ms/step. `get_composed_matrices()` returns the full model's M, Q, G (at d = inf, the
normalised ones); `normalized_matrices()` returns M/√d etc. Only `scripts/train.py`
builds it; `scripts/launcher.py` refuses `backend != "full"`.

**Training helpers (`src/icl/training.py`).** `build_model(model_args, optim_args,
device)` returns `(model, optimizer, log_line)` for either backend (the full path is
exactly `MinimalTransformer` + `initialize_model` + `get_optimizer`).
`get_optimizer(params, optim_args)` and `compute_loss(model, batch, loss_fn, device)`
(the training forward pass, honouring `pred_mode`). Shared by both scripts; not part
of `icl.evaluation`.

**Evaluation (`src/icl/evaluation/`).** One module per job:
- `batch.py`: `preprocess_batch(batch, device)` drops sequences where induction is never
  possible (`filter_batch`) and adds the `ind_possible` mask (`is_trigg & counts > 1`).
  It returns `(batch, PreprocessStats)`; nothing else is precomputed.
- `evaluator.py`: the probing machinery. `EvalContext(model, batch, step, chunk)` is a
  lazy view of the model on the test batch. One forward pass under `inference_mode`
  (train mode restored) runs over the batch `chunk` sequences at a time and keeps only
  `token_loss` (per-position cross-entropy, (B, L)) and `logits_ind` (logits at
  induction-possible positions, (N, V)); no (B, L, V) or (B, L, d) tensor outlives a
  chunk. `ind_index`, `target_ind`, `on/off_target_logits`, `matrices` (from
  `model.get_composed_matrices`) and `normalized_matrices` (divided by √d) are
  computed at most once and only if a probe asks. `outputs` / `attn1/2` are one
  UNCHUNKED `model.full_output` call, for probes that want the attention maps. A
  *probe* is any callable with a `name` taking an `EvalContext` (the `Probe` protocol).
  `Evaluator(scalars=[...], artifacts=[...], chunk)` memoizes the context on `step`:
  `scalars(model, batch, step)` returns `dict[str, float]` (a probe may return a dict,
  merged), `artifacts(model, batch, step)` returns `{(name, group): (data, type)}`, and
  both calls at the same step share one forward pass. Tensors are moved to CPU / numpy
  during packing, so probes stay device-agnostic. `log_artifacts(run, artifacts, step)`
  writes that dict to a TrackLab run (Rewind consumes the same dict directly).
- `scalars.py`: scalar probes (`LossMetric`, `TopKAccuracy`, `TargetProbMass`,
  `LogitStatistics`, and the `M/Q/GammaOrderParameters`, which read
  `normalized_matrices` so they also work at d = inf). Add new metrics here.
- `artifacts.py`: artifact probes; class attributes `group` (artifact subfolder) and
  `atype` (TrackLab serializer, default `tensor`). A dict result saves one artifact per
  key, so a probe that wants a single pickled dict returns `{self.name: payload}`
  (`PerPositionOnOffLogits` does this for `logits/hists_step_N.pkl`; `ComposedMatrices`
  writes `matrices/{M,Q,G}_step_N.npy`). `PerPositionOnOffLogits(fractions)` saves only
  the positions `ceil(f*L) - 1` for `f` in `extra_args.logit_positions` (default
  `[0.5, 0.75, 1.0]`): the pickle is `{'fractions', 'positions', 'on', 'off', 'all'}`
  with one array per saved position. Runs before this change hold one array per
  position (L of them), so old notebooks indexing `lg['on'][l]` need `positions` now.
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
When changing what a run evaluates or saves, change both scripts. It is the only script
that runs `backend=reduced`. Both pass `extra_args.eval_chunk` (default: the training
batch size, which needs less memory than a training step) to the `Evaluator`.
`track_results` records `train_time` and `eval_time` separately, `ms_per_step` of
training alone, and `peak_gpu_mem_gib`.

**Notebooks.** Named `YY_MM_DD_Topic.ipynb`; they read runs back with
`tracklab.ExperimentReader(experiment_name, base_dir='../data')`. Several still import
a `configurations` plotting package that was removed from this repo (the author intends
to move it into `~/research/lib/`); expect those cells to fail until that lands.
