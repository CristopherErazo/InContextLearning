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

# module self-checks (each has an __main__ block)
uv run python -m icl.data
uv run python -m icl.model

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
materialised: the batch carries only the scalars `K` and `V`, and "token `< K`" is the
trigger test everywhere (evaluation and `icl.theory` depend on this). The batch dict also carries
`is_trigg` and `counts` (occurrence count of the token at each position) when
`stats=True`; evaluation recomputes both from `sequence` (ell = counts - 1), so it does
not depend on them.

The generator is loop-free. Because outputs are drawn from `[K, V-1]` they are never
triggers, so no two triggers can be adjacent, and the chain has the closed form
`T_t = b_t ∧ ¬T_{t-1}`, `x_t = output_set[u_{t-1}] if T_{t-1} else u_t` with
`u_t` i.i.d. uniform and `b_t = 1{u_t < K}`; the trigger flags follow from run parity
(a `cummax`) and `counts` from a stable argsort. That invariant is load-bearing — if
outputs ever overlap the trigger set, this derivation is void. Position 0 is drawn from
the chain's stationary law, which is exactly weight `2/(V+K)` on output tokens and
`1/(V+K)` elsewhere. `stats=False` returns only `sequence` / `K` / `V` / `output_set`
and skips the `counts` work; training passes it, evaluation does not.
`device=` draws the batch there directly and `data_args.gen_device` (`"cpu"`, `"cuda"`
or `"auto"` = the training device, the default) is what the scripts pass — benchmark
CPU vs CUDA (`scripts/bench_data.py`, kept on the `code-optimization` branch) before changing it. On the A100 workstation the
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
layer 2 only for the final query. `matrices()` returns the analysis objects as the
plain dict `{"M", "Q", "G"}` (CPU tensors), normalised as in the paper,
`M = Pᵀ WQK1 P / √d` (masked by layer 1's mask), `Q = Eᵀ WQK2 WOV1 E / √d`,
`G = U WOV2 E / √d` (logits `= beta/L Σ M Q G`). That dict is the one matrix
convention of the package: probes, artifacts, `RunData` and
`ReducedTransformer.from_matrices` all use it.

**Reduced backend (`src/icl/reduced.py`, `model_args.backend="reduced"`).** The same
model trained by SGD, rewritten in the coordinates the loss depends on. The logits
depend on the trained weights only through M (L×L), Q, G (V×V), and an SGD step on
(WQK1, WQK2, WOV2) moves them exactly by `M -= eta K_P dM K_P`, `Q -= eta K_E dQ K_R`,
`G -= eta K_U dG K_E`, with fixed Gram matrices `K_P = PPᵀ`, `K_E = EEᵀ`, `K_U = UUᵀ`,
`K_R = E WOV1ᵀ WOV1 Eᵀ` (here M, Q, G are the unnormalised products). `ReducedTransformer` stores everything normalised
(`m = M/√d`, `k = K/d`; m, q, g are the paper's M, Q, Γ), which leaves no d in the forward pass: a step costs
O(B L² V) whatever d_model is, and `d = inf` is `k = I`. `ReducedSGD` applies the
Gram-scaled step with `lr = eta_0 = alpha_lr` (momentum and weight decay map across
exactly; Adam does not, so the backend refuses anything but SGD, and also
`lin_attn=False` and dropout). `model_args.init="full"` composes an initialised
`MinimalTransformer` (the same random draw a full run makes, so old runs can be
reproduced; the d×d weights have to fit once), and `"sample"` draws the Gram matrices
(Bartlett) and initial m, q, g from their joint law directly (any d ≥ max(L, V), and
`model_args.infinite_d=true`). `model_args.ansatz_init=true` then replaces the drawn m, q, g
by their extended-ansatz projection (`ansatz_matrices(measure_order_params(...))`: the
same order parameters, no spread; `model_args.ansatz_keep_spread="QG"` keeps the named
matrices as drawn); `build_model` needs `K=` for it. `tests/test_reduced.py` checks the full and reduced SGD
trajectories agree to 1e-9 in float64 and the sampled init has the full init's
moments. Rerunning `large_d_sweep_1/run_005` (d=2048, L=512) with `init=full` gave
the same early-stop step with metrics equal to ~1e-4 relative (TF32), at 25 vs 356
ms/step. `matrices()` returns m, q, g, the same objects and scale as the full model's
(finite at d = inf). `ReducedTransformer.from_matrices(mats, model_args)` rebuilds the
exact forward pass of any lin_attn run (either backend, any d) from its saved
matrices, with the Gram matrices set to I (so further training is the d = inf
dynamics). Only `scripts/train.py`
builds it; `scripts/launcher.py` refuses `backend != "full"`.

**Reading runs back (`src/icl/runs.py`, `RunData`).** The analysis entry point:
`RunData(experiment, run_id, base_dir)` (or `RunData.last(experiment, base_dir)`) gives
`.config` (typed TrainerArgs as saved), `.metrics` (wide DataFrame indexed by step),
`.steps(group)`, `.load(group, name, step)`, `.matrices(step)`, `.model(step)` (a
`ReducedTransformer.from_matrices`, exact for lin_attn runs) and `.batch(n, seed)` (a
fresh unfiltered batch with the run's V, L, K), `.order_params(step, profile=False,
variances=False)` (every registered order parameter, measured from the saved matrices;
`variances=True` adds `measure_variances`), `.profile(step)` (the `MProfile` probe's
sub-diagonal of M) and `.logit_table(step)` (the
`TriggerLogitTable` probe's `QueryTable`). `step` is an int, `"first"` or `"last"`.
`tests/test_runs.py` checks the round trip model → artifact → `RunData.model` keeps the
logits for both backends, and the order-parameter probes → metrics / `RunData`.

**Effective model (`src/icl/theory/`, paper/scratch/extended_ansatz.tex).** Conventions:
positions are the paper's `mu = code position + 1`, `ell` = earlier occurrences of the
query, rows exist only at trigger queries (ell >= 0), and logit vectors are in the
canonical layout of `logit_blocks(K, V)`: `[0,K)` triggers, `[K,2K-1)` outputs of the other
triggers (by trigger), `[2K-1,V-1)` the rest (by id), `V-1` the target.
- `query_table.py` (ansatz-independent): `QueryTable`, a dict of tensor columns with
  one row per trigger query plus metadata (`num_rows`, `select`, `groups`,
  `concatenate`, `to_numpy` / `from_numpy`); `logit_blocks(K, V)` (column slices
  `triggers`, `other_outputs`, `rest`, `target`, `non_target`); `canonical_permutation`;
  `trigger_queries`; `logit_table(model, batch, mus, chunk)` (the model's canonical
  logits at trigger queries, with `mu`, `ell`, `sequence_index`);
  `cluster_cross_entropy(table)` (`count` and mean `cross_entropy` per (mu, ell),
  differentiable).
- `ansatz.py`: `ORDER_PARAMS = {name: (matrix, support(L, V, K))}`, the single
  definition of the order parameters (value = mean over the support);
  `measure_order_params`, `ansatz_matrices` (inverse of the former), `support_sizes`,
  and `ansatz_logits(vars, op, beta, L)` (eq. logit_classes_explicit; a missing
  parameter counts as 0). The variance ansatz
  (paper/scratch/2026-10-05-0036_variance-profile-ansatz-explicit.tex): `VARIANCES`, the
  eight blocks `var_M_on, var_M_off, var_Q_on, var_Q_T, var_Q_N, var_G_on, var_G_T,
  var_G_N` (variance around the block's own mean; `var_M_on` around the smooth profile,
  half the mean squared neighbouring difference), and `POOLED_VARIANCES` `var_M, var_Q,
  var_G` (pooled within-block, not the raw matrix variance), which stand in for any
  block of their matrix without its own key (`block_variance(op, name)`); a missing
  variance counts as 0, i.e. no noise. `measure_variances(matrices, K)` (the 8 + 3 and
  `var_M_on_raw`, the sub-diagonal around M_on), `variance_sizes`, `DIAGNOSTIC_MEANS`
  `Q_N, G_N` (zero in the ansatz, measured by `measure_diagnostic_means`),
  `ansatz_matrices(op, ..., noise=generator)` (adds i.i.d. Gaussian block noise: a draw
  of the variance ansatz), and `LEVELS` S0..S5 with `at_level(op, level)` (the keys each
  simplification level keeps; `"S5"` = `SIGNAL` = M_on, Q_on, G_on).
- `variables.py`: the Table 1 variables (paper names `N, F, R, W, P` and `U_bar, W_bar,
  P_bar`) from three sources, all `QueryTable`s ready for `ansatz_logits`:
  `measure_variables(batch, mus)` (exact, rows aligned one to one with `logit_table` on
  the same batch); `sample_variables(mu, ell, V, K, num_samples, counts="poisson" |
  "multinomial", generator)` (the scratch file's sampling protocol at fixed (mu, ell),
  mu/ell ints or per-row tensors; padded + masked, pair counts by `searchsorted`); and
  `mean_variables(mu, ell, V, K)` (Table 2 means, so `ansatz_logits` of it is the mean
  logit vector). `tests/test_sampler.py` checks every Table 2 mean / variance and the
  listed covariances (Poisson counts reproduce them exactly) and the multinomial budget.
  One more column, for the logit noise only: `Y_bar` (n, V-K-1), the ordered pairs
  c_tr (c_tr - 1) of occurrences of each output that follow its trigger (0 for the plain
  tokens); sampled it is its mean given the count, c (c - 1) / 4, so no draw is added and
  the other columns keep their random stream.
- `variables.py` additions for the noise: `U_triggered` (bool, the shape of `U_keys`: which
  keys of an output follow its trigger, the predecessor type; always measured, sampled with
  `sample_variables(..., triggered=True)`, drawn after everything else so the other columns
  keep their stream), and the non-trigger-query tables `measure_nontrigger_variables` /
  `sample_nontrigger_variables` (per non-trigger token, layout `nontrigger_blocks(K, V)`:
  outputs by trigger, then the rest).
- `terms/` (the term layer, paper/appendix/appendix.tex sections Ansatz and Effective Loss;
  plan and audit in paper/scratch/2026-10-08-1103_plan-pruned-effective-model.md and
  2026-10-08-1119_pruned-effective-model.tex): every boxed equation of the logit moments as
  named terms, at every level of conditioning: `exact_trigger_terms` / `exact_nontrigger_terms`
  (exact given the sequence, eq. cov_exact split by mechanism; tests and audit),
  `trigger_terms(variables, op, beta, L)` / `nontrigger_terms` (given S_mu, eqs. pairs_trigger,
  pairs_nontrigger), `trigger_terms_ell(mu, ell, op, beta, L, V, K)` (given ell: means, spread of
  the means, mean noise) / `nontrigger_terms_mu`. Each returns a `TermTable`: terms with a name,
  the appendix label and tags (pair of classes "on,on", "T,T", "T,T'", "on,T", "b,b", "on,b",
  "b,T", "b,b'" and the means "on", "T", "b"; channel mean / spread / attention / readout;
  column; mechanism incoherent, source_coherent, token_coherent, a_incoherent, a_coherent,
  mean_amplitude; key set; detail; params), `total(pair, keep)`, `moments`, `aggregate(by)`,
  `select`, `class_mean`; `keep` = term names, a predicate on `Term`, or a preset of `PRESETS`
  ("full", "means", "no_noise", "no_spread", and the pruned models of Phase 6, nested:
  "counting" < "pruned" < "pruned_late" < "compact" < "minimal", whose dropped names live in
  `terms/presets.py` (generated by `scripts/presets.py --write`, validated at run_001; they
  prune trigger-query terms only, levels S and ell, and keep every non-trigger-query term). b-b' covariances
  are `Bilinear` (sums of outer products; `.dense()`, `.class_sums(classes)`). One definition of
  the pieces (`forms.py`: products of linear forms in the statistics) serves the S and ell
  levels: given ell a piece is E X E Y + Cov(X, Y) from tables moments/cov, so each ell term is
  the conditional mean of the S term of the same name. Defaults differ from the appendix as
  written by the approved audit items b1, b2 (no same-source pairs sigma = sigma' in the
  token-coherent terms); `as_written=True` gives the appendix as written. Options: `pairs=`
  (compute only some pairs), `Lambda="mu" | "n-2ell"`, `background="exact" | "leading"`,
  `classes_only=True` (ell level, one output and one plain column). Without `U_triggered` the
  statistics use their mean over the predecessor types given the keys (exact for every term).
  `tests/test_terms.py`: S and ell levels equal the literal appendix transcriptions of
  `tests/terms_oracle.py` (as written, 1e-12), the exact level the brute second moments, S vs
  the exact covariance (variances within 2.5%, covariances within MC error), ell = mean of S over
  the sampler term by term, ell vs the true chain (8%), keep / presets, gradcheck, GPU.
  `scripts/audit_theory.py` is the Phase 2 audit (results in `data/audit/`).
  `scripts/term_hierarchy.py` is the Phase 4 hierarchy at run_001 (5 snapshots; trained net,
  128 ansatz networks, ell and S levels; ranking and greedy pruning, `--from-csv` redoes only
  the pruning; results in `data/hierarchy/`, ~15 min on one GPU).
  `terms/symbolic.py` (sympy, not re-exported by `icl`): `symbolic_terms_ell(pairs, background)`
  is every ell term as a sympy expression in `SYMBOLS` (same names and tags; a b column of class
  s, s2 for b' of "b,b'"), built from the same pieces on symbols (`forms.moment_tables` is shared
  with `GivenEll`); with `background="exact"` it equals `trigger_terms_ell` term by term
  (`tests/test_symbolic.py`). `normalized_terms_ell()` writes each term in units of
  h0 = (beta/L) G_on Q_on Phi1 (h0^2 for covariances) in the ratios of `RATIOS` (r_M = M_off mu /
  Phi1, s_Moff, s_Mon, r_Q, s_Qon, s_QT, r_G, s_Gon, s_GT, s_GN, d2, dPsi), mu, ell, Lambda,
  rho = K/V (leading background, n = mu); `monomials(expr)` groups it by powers of mu and the
  ratios; `ratio_values(op, window, mu, V, K)` gives the numbers. `symbolic_terms_S(query)` is the level given S: every
  term of `trigger_terms` / `nontrigger_terms` as a formula in the statistics (`S_SYMBOLS`),
  equal row by row; `scripts/report_tables.py` renders it and the Phase 5-6 tables as LaTeX
  (`data/report/`) for the final report paper/scratch/2026-10-08-1907_pruned-effective-model.tex. `scripts/power_counting.py` is
  the Phase 5 power counting (estimates and coefficients at run_001, exponents in L and V from
  the runs of experiment `power_counting` (launched by `shell/power_counting_runs.sh`: run_001's
  setup with one of V, L changed), agreement with the Phase 4 pruning; results in
  `data/hierarchy/phase5_*`, ~1 min CPU; `--no-runs` skips the scaling). `scripts/presets.py`
  is Phase 6: builds the nested presets from the Phase 4/5 tables, validates them (tolerance,
  rms z against the ansatz networks and the trained net, power counting) and gives the
  effective loss per preset for information (`data/hierarchy/phase6_*`; `tests/test_presets.py`).
- `noise.py` (the logit fluctuation; on the term layer since 2026-10-08): the logits are the
  mean logits + xi, xi Gaussian given the sequence. `noise_variances(vars, op, beta, L, keep=None,
  covariances=False)` gives per trigger query `target` Var h^on, `trigger_common` V_c^T (shared
  by the K triggers), `trigger_individual` V_i^T, `non_target` Var h_b (n, V-K-1) and
  `target_trigger` Cov(h^on, h_tau), given S_mu (`trigger_terms`); `covariances=True` adds
  `target_non_target`, `non_target_trigger` and `non_target_pairs` (a Bilinear), which the
  closure does not use. `closure_variances(table, keep)` reads the five from any trigger
  TermTable. `sample_logits(vars, op, beta, L, generator, normals=None, keep=None)` adds a
  Gaussian draw (target and common trigger part jointly, the rest independent); fixed `normals`
  (n, V+1: the canonical layout, then the common part) are common random numbers.
  `non_trigger_loss(mu, op, beta, L, V, K, keep=None, Lambda="n-2ell")` is L_N(mu) to second
  order, (1/2)[(1/V) sum_c Var xi_c - Var xi_bar] from `nontrigger_terms_mu` with every pair
  (`terms.nontrigger_second_order`), the token rate p_T (mu - 2). Without variances all three
  reduce exactly to no noise / `ansatz_logits` / log V. The deterministic key sums D, G1, K1, K2,
  J2, X1, X2 live in `sums.py`.
- `effective.py`: `EffectiveLoss(V, L, K, beta, method="mean" | "mc", mus, num_samples,
  counts="multinomial", ell_tol, seed, device)` (or `.from_config(run.config, ...)`), the
  population loss `(1 - q_T) log V + q_T mean_mu sum_ell Poisson(ell; p_T mu) CE(mu, ell)`
  with q_T = K/(V+K): "mean" takes CE at the mean logits, "mc" averages over samples
  drawn once in the constructor (common random numbers, so the loss is smooth). Calling
  it on `{name: value}` gives a differentiable 0-d tensor; `value_and_grad` returns the
  gradient for every registered parameter; `breakdown` gives `loss`, `loss_trigg`,
  `loss_non_trigg` and a per-(mu, ell) `clusters` table, and `trigger_loss(clusters, ell)`
  is the analogue of `LossMetric(positions="trigg", ell=...)`. `tests/test_effective_loss.py`
  checks it against the measured loss of a model with exact ansatz matrices (total to
  0.5%, ell >= 1 to 3%), `gradcheck`, and the GPU path. Selection conditions (`None`, k,
  `[k, ...]`, `">=k"`) are one helper, `condition_mask` in `query_table.py`, shared by
  `QueryTable.select`, `trigger_loss` and `LossMetric`.
  With variance keys in `op` (block or pooled; any present switches the noise on, so
  without them both methods are bit-identical to the signal-only loss) the expectation is
  also over the logit noise of `noise.py`: the non-trigger part is the mean over `mus` of
  `non_trigger_loss` (`loss.non_trigger_loss(op)`, in `breakdown` as `loss_non_trigg`);
  "mc" adds one draw of the noise per sampled row (`sample_logits` with normals frozen in
  the constructor); "mean" keeps the mean logits and applies the Gaussian closure
  `closure_cross_entropy(logits, variances, K)` (K e^{h_T + xi_c + V_i/2} for the
  triggers, sum_b e^{h_b + V_b/2} for the non-targets, the (target, common trigger) pair by
  a trapezoid grid of standard normals, `QUADRATURE_SPACING` 0.2 on [-8, 8]; Gauss-Hermite
  is badly off once the noise is large), with the covariance given ell in closed form from
  `trigger_terms_ell(..., classes_only=True)`: the mean noise plus the spread of the means
  Cov(hbar | ell) (eq. cov_total; since 2026-10-08; `noise_samples` is no longer used).
  Options: `keep` (prunes the terms of the logit moments, both methods), `spread` ("always",
  the default since 2026-10-08: the closure with the spread also without variances, so the
  signal-only "mean" loss is no longer CE at the mean logits and signal-only flows changed;
  True: the spread only when there are variances; False: mean noise only / CE at the mean
  logits),
  `Lambda` ("mu" | "n-2ell", the rate of the ell moments of "mean"). `value_and_grad` also
  returns the gradient of every variance key given. Cost with variances (V=128, L=256, 64
  mus): "mean" 0.28 s per `value_and_grad` on the CPU, 0.09 s on the GPU (L=512, every mu:
  1.9 s / 0.11 s).
  `tests/test_effective_noise.py`: no variance = signal only, the noise raises the loss,
  "mean" and "mc" agree on its cost (5%), the closure against a direct Monte Carlo of
  its integral, `gradcheck` in means and variances, the non-trigger part against the
  model on `noise=` draws. Against the model the trigger part needs ~100 networks (one
  network's trigger loss scatters by ~0.1 nats: its K query rows of Q are quenched); over
  128 the noise cost agreed to 5% (total 0.0465 +- 0.0010 vs 0.0485-0.049, trigger part
  0.122 +- 0.007 vs 0.131-0.134).
  `integrate(loss, initial_order_params, rates, steps, record_steps, method="LSODA")`
  runs `d theta_i/d step = -rates[i] d loss/d theta_i` with `scipy.integrate.solve_ivp`
  (time in SGD steps; the rates are the caller's, e.g. derived from the projection of
  ReducedSGD); parameters absent from `rates` stay frozen at their initial value (0 if
  absent there too); gradients by autograd, so any callable `{name: tensor} -> tensor`
  works. Returns a DataFrame indexed by `step` with the metric names + `loss`, ready to
  overlay on `RunData.metrics`. All parameters at 0 is a fixed point. Variance keys flow
  multiplicatively, `d var/d step = -rates[var] var d loss/d var` (0 stays 0; one without
  a rate is passed to the loss as given; each given variance is a column).
  `exact_rates(config, names)` gives the rates of the d = inf SGD dynamics of a run
  (`RunData.config`): eta_0 = alpha_lr (ReducedSGD's step), kappa_M = kappa_Gamma = 1,
  kappa_Q = sigma_0^2 (k_R of the frozen WOV1); a mean eta_0 kappa_X / n (support size),
  `M_profile` eta_0 per entry, a variance 4 eta_0 kappa_X / n (pooled: n = all entries of
  the matrix's blocks); plain SGD only. `tests/test_flow.py` also checks the closed-form
  variance flow and that one ReducedSGD step at d = inf moves every block mean by exactly
  `-exact_rates x` its batch gradient.
  `asymptotic_loss(V, L, K)` is the manuscript's L^infty and `learning_time(loss, V, L, K,
  fraction=0.2)` the first step (interpolated) at which a loss series has gone `fraction`
  of the way from log V to it: the learning time used for runs, controls and flows alike
  (top-1 accuracy is meaningless without the spread of the entries: the argmax of
  deterministic ansatz logits only sees their signs). `tests/test_flow.py`
  checks a quadratic loss against its closed form, freezing, and descent.
- **Profile** (paper/scratch/2026-10-03-1539_profile-ansatz.tex): the previous-token
  diagonal can be a tensor, `order_params["M_profile"]` of length L-1 in the
  `M.diagonal(-1)` convention (entry i = M[i+1, i]); it replaces M_on in
  `ansatz_logits` and `ansatz_matrices` (M_on stays the logged scalar mean; the key
  `PROFILE` names it). The logits then read the profile at the keys of each occurrence:
  `measure_variables` / `sample_variables` return "N_keys" (witness keys), "F_keys"
  (free target keys), "U_keys" ((n, V-K-1, c) keys of each non-target token), as code
  positions 1..mu-2 padded with 0; `mean_variables` has no keys, so it uses the count
  times the mean of the profile over the query's keys. Constant profile == scalar
  ansatz on every source. `EffectiveLoss` works unchanged (differentiable in the
  profile), `value_and_grad` returns a tensor gradient for it, `integrate` can move it
  (rate per entry or shared; trajectory in column "M_profile", "M_on" = its mean) or move
  it inside a family, `profile_family=(function, theta0)`: theta follows the least-squares
  projection of the entry-wise flow, `-rate (J^T J)^{-1} d loss/d theta` (column
  "M_profile_params"; the constant family == the M_on flow at rate/(L-1)), and
  `RunData.order_params(step, profile=True)` adds the measured profile.
  `tests/test_profile.py`: exactness with a random profile, constant profile == scalar,
  keys vs counts, sampled witness sums vs (ell mean, ell var) of the profile, gradcheck,
  flow.
Only `ansatz.py`, `variables.py`, `sums.py` and `terms/` know the ansatz; extending it = register the
parameter + add its terms (and variables). `tests/test_theory.py` runs the model on
`ansatz_matrices(op)` and requires `ansatz_logits(measure_variables(...))` to match to
1e-12 at every trigger query: keep it passing when the ansatz changes.
`tests/test_golden.py` recomputes ~500 golden values of every public usage on fixed inputs
(`tests/golden_theory.py`, fixture `tests/golden/theory_golden.pt` with run_001's order
parameters) and fails on any change except the agreed ones of `EXPECTED_CHANGES`, whose new
values are pinned in `tests/golden/theory_golden_changes.pt` (`golden_theory.py --changes`).
`tests/test_variances.py` checks the block-variance estimators on `noise=` draws with
known variances (the sub-diagonal around a smooth profile included), the pooled
variances, `at_level` and the order-parameter probes. `tests/test_noise.py` checks the
noise formulas against the exact covariance given the sequence (eq. cov_exact, computed
per row; within 3%, the covariance 4%, the non-trigger loss 2%, at a mid-training and a
trained-like point), that exact covariance against the model on `noise=` draws, no
variance = no noise, the initialisation floor V_0 and the `sample_logits` covariance.

**Training helpers (`src/icl/training.py`).** `build_model(model_args, optim_args,
device)` returns `(model, optimizer, log_line)` for either backend (the full path is
exactly `MinimalTransformer` + `initialize_model` + `get_optimizer`).
`get_optimizer(params, optim_args)` and `compute_loss(model, batch, loss_fn, device)`
(the training forward pass, honouring `pred_mode`). Shared by both scripts; not part
of `icl.evaluation`.

**Evaluation (`src/icl/evaluation/`).** One module per job:
- `batch.py`: `preprocess_batch(batch, device)` moves the batch to `device` and returns
  `(batch, PreprocessStats)` (sequences, fraction of trigger queries, fraction with
  ell >= 1). The batch is NOT filtered (it used to drop sequences with no
  induction-possible position, which biased the logged `loss` and the ell = 0
  frequency), so `loss` estimates the population loss. There is no induction mask any
  more: probes select positions by `is_trigger` and `ell`.
- `evaluator.py`: the probing machinery. `EvalContext(model, batch, step, chunk)` is a
  lazy view of the model on the test batch. One forward pass under `inference_mode`
  (train mode restored) runs over the batch `chunk` sequences at a time and keeps only
  `token_loss` (per-position cross-entropy, (B, L)) and `trigger_logits` (logits at
  every trigger query, (N, V), raw token order); no (B, L, V) or (B, L, d) tensor
  outlives a chunk. Set eagerly: `is_trigger` and `ell` ((B, L): earlier occurrences of
  the query token, from the sequences). Lazy: `trigger_index`, `trigger_target`,
  `trigger_ell`, `matrices` (`model.matrices()`), each computed at most once and only if
  a probe asks. `outputs` / `attn1/2` are one
  UNCHUNKED `model.full_output` call, for probes that want the attention maps. A
  *probe* is any callable with a `name` taking an `EvalContext` (the `Probe` protocol).
  `Evaluator(scalars=[...], artifacts=[...], chunk)` memoizes the context on `step`:
  `scalars(model, batch, step)` returns `dict[str, float]` (a probe may return a dict,
  merged), `artifacts(model, batch, step, schedule="artifact")` returns `{(name, group):
  (data, type)}` of the artifact probes whose class attribute `schedule` (default
  `"artifact"`) matches; `"scalar"` ones are saved at every scalar evaluation (both
  scripts do so; `evaluator.artifact_schedules` says whether any probe asks), and
  all calls at the same step share one forward pass. Tensors are moved to CPU / numpy
  during packing, so probes stay device-agnostic. `log_artifacts(run, artifacts, step)`
  writes that dict to a TrackLab run (Rewind consumes the same dict directly).
- `scalars.py`: scalar probes. `LossMetric(name=None, positions="all"|"trigg"|"non_trigg",
  ell=None|k|[k, ...]|">=k")` (ell only with "trigg"; default names `loss`,
  `loss_trigg`, `loss_trigg_ell0`, `loss_trigg_ell_ge1`, ...; NaN when no position of
  the test batch matches, e.g. an ell too large). `TopKAccuracy` and `TargetProbMass`
  are the in-context metrics: trigger queries with ell >= 1 only, NaN if there are
  none. `OrderParameters(names=None)` logs the block means from `matrices` (finite at
  d = inf; default every entry of `icl.theory.ORDER_PARAMS`, `names` may add `Q_N`,
  `G_N`), `BlockVariances(names=None)` the block variances (`measure_variances`).
  `order_parameter_probes(names)` turns `extra_args.log_order_params` (groups `"means"`,
  `"variances"`, `"profile"` and/or single metric names; default all three; `[]` = none)
  into (scalar probes, artifact probes); both scripts use it. Add new metrics here.
- `artifacts.py`: artifact probes; class attributes `group` (artifact subfolder),
  `atype` (TrackLab serializer, default `tensor`) and `schedule` (default `"artifact"`).
  A dict result saves one artifact per key, so a probe that wants a single pickled dict
  returns `{self.name: payload}` (`ComposedMatrices` writes one `matrices/matrices_step_N.pkl` holding
  `model.matrices()` as a dict of numpy arrays; `TriggerLogitTable(fractions)` writes
  `logits/table_step_N.pkl`, the `logit_table` of the test batch at
  `mu = ceil(f*L)` for `f` in `extra_args.logit_positions`, as a numpy dict; `MProfile`
  writes `profile/M_profile_step_N.npy`, the sub-diagonal of M, at every scalar evaluation:
  `schedule = "scalar"`).
- `schedule.py`: `get_evaluation_times(extra_args)` turns `n_prints` / `n_prints_model`
  / `print_scale` into two step sets spanning `[0, total_steps]` inclusive. Evaluation
  at step `s` sees the weights before the s-th update, so `total_steps` is the final
  model; both loops evaluate it after the last update when it is in the schedule.

**Launcher (`scripts/launcher.py`).** Composition root for controllable runs.
`build_controller` wires three closures — `train_step_fn`, `eval_fn`, `eval_artifacts_fn`
— plus a TrackLab run into `rewind.TrainerController`, then sets
`controller.eval_schedule` / `eval_artifacts_schedule` to the computed step sets (the
artifact one also holds the scalar steps when a probe has `schedule = "scalar"`, and
`eval_artifacts_fn` picks the probes by step). The
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
`extra_args.stop_at_loss` stops on the loss like `stop_at_accuracy` does on the
accuracy. `scripts/L_sweep_controls.py` (ansatz-init reruns of the d = inf `L_sweep_reduced`
runs) and `scripts/L_sweep_flows.py` (effective flows of the signal / trigger / extended
variants, to `data/effective_L_sweep/`, rates from `exact_rates`; the sigma_0 = 0.5 flows
written before 2026-10-05 had Q rates 4x too large) feed
`notebooks/26-10-04_L_sweep_effective.ipynb`. `scripts/L_sweep_setup_flows.py` runs the
effective flows of the setups of `notebooks/26-10-05_Ansatz-check.ipynb` (signal, means, + noise,
+ entry-wise profile) over the L of `L_sweep_B1024`, from the typical initial values
(`--init typical`, seed-free: means s_X/sqrt(n), G_on with the sign `--signs`, variances s_X^2,
flat profile; reference T* = median over the 30 seeds) or the step-0 order parameters of a run
(`--init measured --seeds`), in doubling chunks until the loss is half-way to L^infty, solver
"auto" (LSODA up to 300 moving entries; the noise setups are stiff, RK45 crawls there; RK45 for
the large profiles). Each flow is a TrackLab run (default experiment `setup_flows_B1024`) that
RunData reads: `config.flow_args` (the `FlowArgs` of `config.py`, empty for training runs),
`metrics` (order parameters + `loss`), `profile(step)`, results `T_f20`, `T_f50` and
`T_reference_f20/50`; it resumes. (`data/effective_setups_B1024/` holds the older pickle output
of the seed-13 measured sweep.) RunData assumes a `TrainerArgs` config and training artifacts;
`FlowArgs` is a stopgap until RunData is generalised to such non-training runs.
`notebooks/26-10-05_L512_ansatz_study.ipynb` compares empirical and ansatz logits of
`data/L512_ansatz_study` (a dense rerun of the seed-13 L = 512 run) and the effective loss
per level. In `icl.theory`, index the profile by key positions with `ansatz._take`
(index_select), not `vector[index]`: the backward of advanced indexing on CUDA is 50-60x
slower with millions of repeated keys.
`track_results` records `train_time` and `eval_time` separately, `ms_per_step` of
training alone, and `peak_gpu_mem_gib`.

**Branches.** `main` is kept lean: `src/`, `tests/`, `scripts/{train,launcher,dash}.py`,
`shell/{submit,train,workstation_sweep,cluster_status,leonardo_sweep}.sh` and only
`notebooks/template.ipynb`. `code-optimization` is the archive with the full history of
dated notebooks, benchmarks and extra shell helpers; `new-ansatz` is the active
development branch, created from `main`.

**Notebooks.** Only `template.ipynb` lives on `main`; dated ones are named
`YY_MM_DD_Topic.ipynb` (see `code-optimization`). They read runs back with
`tracklab.ExperimentReader(experiment_name, base_dir='../data')`. Several still import
a `configurations` plotting package that was removed from this repo (the author intends
to move it into `~/research/lib/`); expect those cells to fail until that lands.
