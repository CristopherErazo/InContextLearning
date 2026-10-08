"""The effective model's population loss, as a differentiable function of the
order parameters.

At a non-trigger query every ansatz logit is 0, so the cross-entropy is log V.
At a trigger query at mu with ell earlier occurrences it is
CE(mu, ell) = E[logsumexp(h) - h_target], h = ansatz_logits of the variables.
With q_T = K / (V + K) the probability that a position holds a trigger and
ell ~ Poisson(p_T mu), p_T = 1 / (V + K),

    loss = (1 - q_T) log V + q_T * mean over mu of sum_ell P(ell | mu) CE(mu, ell),

the mean over mu running over every position 1..L, or over the chosen `mus`.
The expectation in CE is taken one of two ways:

- method="mean": CE at the mean logit vector, ansatz_logits(mean_variables);
- method="mc":   the average over `num_samples` draws of `sample_variables`
                 per (mu, ell), drawn once in the constructor and reused, so the
                 loss is a smooth deterministic function of the order parameters.

With block variances in the order parameters (the variance ansatz, `noise.py`)
the logits are h + xi, xi Gaussian given the sequence, and the expectation is
also over xi: the non-trigger part becomes the mean over mu of
`non_trigger_loss` (log V + half the spread of the logit vector), and in CE

- method="mc":   each sampled row adds one draw of xi (`sample_logits`, the moments
                 given S_mu), with normals drawn once in the constructor;
- method="mean": the Gaussian closure at the mean logits: the K individual trigger
                 parts and the non-targets by e^{V/2}, the target and the common
                 trigger part by quadrature, with the covariance given ell of the
                 term layer (`terms.trigger_terms_ell`, closed form): the mean noise
                 plus the spread of the means Cov(hbar | ell) (eq. cov_total; audit item
                 c2). By default (spread="always") the closure is used also without
                 variances, the spread of the means alone.

Without any variance key "mc" is exactly the signal-only loss (CE over the sampled
variables) and "mean" with spread=False is CE at the mean logits. `keep` prunes the terms
of the logit moments (`icl.theory.terms`), `Lambda` sets the occurrence rate of the
ell-level moments ("mu": p_T mu, "n-2ell": p_T (mu - 2 - 2 ell)).

`integrate` runs the gradient flow of such a loss in the order parameters
(the variances multiplicatively), with the rates given by the caller;
`exact_rates` gives those of the d = inf SGD dynamics of a run.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from scipy.stats import poisson

from .ansatz import ORDER_PARAMS, POOLED_VARIANCES, PROFILE, VARIANCES, ansatz_logits, support_sizes, variance_sizes
from .noise import (CLOSURE_PAIRS, _target_and_common, closure_variances, expand_classes, mean_logits, non_trigger_loss,
                    sample_logits)
from .terms import trigger_terms_ell
from .query_table import QueryTable, cluster_cross_entropy, condition_mask, logit_blocks
from .variables import COUNT_LAWS, mean_variables, sample_variables

METHODS = ("mean", "mc")
VARIANCE_KEYS = (*VARIANCES, *POOLED_VARIANCES)
# the closure's (target, common trigger) integral: trapezoid rule on a grid of standard normals in [-8, 8]^2, which
# converges exponentially for these softplus-like integrands (Gauss-Hermite does not once the noise is large: 16 nodes
# were off by 1e-2 at moderate spreads, this grid by 1e-7, and by 3e-4 with standard deviations of 40)
QUADRATURE_SPACING, QUADRATURE_RANGE = 0.2, 8.0


def _has_noise(order_params: dict) -> bool:
    return any(name in order_params for name in VARIANCE_KEYS)


class EffectiveLoss:
    """The effective population loss for a task (V, L, K) and a model with
    inverse temperature beta.

        loss = EffectiveLoss.from_config(run.config, method="mc", mus=range(16, 257, 16))
        loss(order_params)                       # 0-d tensor, differentiable in the values
        loss.value_and_grad(order_params)        # (float, {name: d loss / d name})
        loss.breakdown(order_params)             # loss per part and per (mu, ell) cluster

    `order_params` is {name: float or 0-d tensor} over the names of ORDER_PARAMS,
    plus optionally "M_profile" and the block variances (VARIANCES, POOLED_VARIANCES);
    a missing one counts as 0. ell is truncated where the Poisson tail falls
    below `ell_tol` (the kept probabilities are renormalised). For method="mc"
    the cached variables take about (number of clusters) x num_samples x V
    numbers: thin out `mus` for long sequences. method="mean" with variances takes
    the moments given ell in closed form (`noise_samples` is no longer used and is
    kept for compatibility).

    keep: prunes the terms of the logit moments (term names, a predicate on
    `terms.Term` or a preset of `terms.PRESETS`); None keeps all. spread (method="mean"):
    "always" (default since 2026-10-08) uses the closure with the spread of the means
    Cov(hbar | ell) also without variances (the signal-only loss is then not CE at the
    mean logits); True includes the spread only when the closure is used anyway (with
    variances), so that without variances the loss is CE at the mean logits; False is
    the closure with the mean noise only, or CE at the mean logits without variances. Lambda: "mu" or "n-2ell", the rate of
    the ell-level moments of method="mean" (the means too, if not "mu").
    """

    def __init__(self, V: int, L: int, K: int, beta: float, method: str = "mean", mus=None,
                 num_samples: int = 256, counts: str = "multinomial", ell_tol: float = 1e-6,
                 seed: int = 0, dtype=torch.float64, device=None, noise_samples: int = 32, keep=None,
                 spread: bool | str = "always", Lambda: str = "mu"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if spread not in (True, False, "always"):
            raise ValueError(f"spread must be True, False or 'always', got {spread!r}")
        if counts not in COUNT_LAWS:
            raise ValueError(f"counts must be one of {COUNT_LAWS}, got {counts!r}")
        self.V, self.L, self.K, self.beta, self.method = V, L, K, beta, method
        self.dtype, self.device = dtype, device
        self.num_samples, self.noise_samples, self.counts, self.seed = num_samples, noise_samples, counts, seed
        self.keep, self.spread, self.Lambda = keep, spread, Lambda
        self.trigger_probability = K / (V + K)
        mus = torch.arange(1, L + 1) if mus is None else torch.as_tensor(list(mus)).reshape(-1)
        if (mus < 1).any() or (mus > L).any():
            raise ValueError(f"mus must lie in 1..L = 1..{L}")
        self.mus = mus.long()

        # the (mu, ell) clusters, sorted by mu then ell, and P(ell | mu)
        cluster_mu, cluster_ell, probability = [], [], []
        for mu in self.mus.tolist():
            mean_count = mu / (V + K)
            ell_max = int(poisson.isf(ell_tol, mean_count)) if mean_count > 0 else 0
            ells = torch.arange(ell_max + 1)
            weights = torch.as_tensor(poisson.pmf(ells.numpy(), mean_count))
            cluster_mu.append(torch.full_like(ells, mu))
            cluster_ell.append(ells)
            probability.append(weights / weights.sum())
        self.clusters = QueryTable({"mu": torch.cat(cluster_mu), "ell": torch.cat(cluster_ell),
                                    "probability": torch.cat(probability).to(dtype)})

        if method == "mean":
            self.variables = mean_variables(self.clusters["mu"], self.clusters["ell"], V, K,
                                            dtype=dtype, device=device)
        else:
            generator = torch.Generator(device or "cpu").manual_seed(seed)
            self.variables = self._sample(num_samples, generator)
            # one draw of the logit noise per row (the canonical layout, then the common trigger part)
            self.normals = torch.randn(self.variables.num_rows, V + 1, generator=generator, dtype=dtype,
                                       device=device)

    def _sample(self, num_samples: int, generator) -> QueryTable:
        """`num_samples` draws of the variables per cluster, cluster after cluster."""
        return sample_variables(self.clusters["mu"].repeat_interleave(num_samples),
                                self.clusters["ell"].repeat_interleave(num_samples), self.V, self.K,
                                counts=self.counts, generator=generator, dtype=self.dtype, device=self.device)

    @classmethod
    def from_config(cls, config, **options) -> "EffectiveLoss":
        """From a run's TrainerArgs (e.g. `RunData.config`): V, L, K and beta."""
        return cls(config.model_args.vocab_size, config.model_args.seq_len, config.data_args.K,
                   config.model_args.beta, **options)

    # ---- the loss -------------------------------------------------------------

    def cluster_cross_entropy(self, order_params: dict) -> torch.Tensor:
        """(number of clusters,) CE(mu, ell), in the order of `self.clusters`."""
        if self.method == "mean" and (_has_noise(order_params) or self.spread == "always"):
            return self._closure_cross_entropy(order_params)
        if self.method == "mc":
            logits = (sample_logits(self.variables, order_params, self.beta, self.L, normals=self.normals, keep=self.keep)
                      if _has_noise(order_params) or self.keep is not None
                      else ansatz_logits(self.variables, order_params, self.beta, self.L))
        else:
            logits = (self._ell_terms(order_params, means_only=True)[1] if self._custom_means()
                      else ansatz_logits(self.variables, order_params, self.beta, self.L))
        table = QueryTable({"logits": logits, "mu": self.variables["mu"], "ell": self.variables["ell"]})
        return cluster_cross_entropy(table)["cross_entropy"]

    def _custom_means(self) -> bool:
        """Whether the mean logits differ from ansatz_logits of the cached variables."""
        return self.keep is not None or (self.method == "mean" and self.Lambda != "mu")

    def _ell_terms(self, order_params: dict, means_only: bool = False):
        """The term layer given ell at the clusters (one output and one plain token): the
        five closure moments with the kept terms (no spread if spread=False) and the mean logits."""
        pairs = ("on", "T", "b") if means_only else ("on", "T", "b", *CLOSURE_PAIRS)
        table = trigger_terms_ell(self.clusters["mu"], self.clusters["ell"], order_params, self.beta, self.L, self.V,
                                  self.K, Lambda=self.Lambda, dtype=self.dtype, device=self.variables["N"].device,
                                  pairs=pairs, classes_only=True)
        kept = {name for name in table.resolve(self.keep) if self.spread or table[name].channel != "spread"}
        logits = (mean_logits(table, kept) if self._custom_means()
                  else ansatz_logits(self.variables, order_params, self.beta, self.L))
        if means_only:
            return None, logits
        variances = closure_variances(table, kept)
        variances["non_target"] = expand_classes(variances["non_target"], self.K, self.V)
        return variances, logits

    def _closure_cross_entropy(self, order_params: dict) -> torch.Tensor:
        """method="mean" with variances: the closure at the mean logits, with the covariance
        given ell (mean noise + spread of the means) of the term layer."""
        variances, logits = self._ell_terms(order_params)
        return closure_cross_entropy(logits, variances, self.K)

    def non_trigger_loss(self, order_params: dict):
        """The cross-entropy at the non-trigger queries, averaged over `mus`: log V
        without variances, else the mean of `noise.non_trigger_loss` (a tensor)."""
        if not _has_noise(order_params):
            return math.log(self.V)
        return non_trigger_loss(self.mus, order_params, self.beta, self.L, self.V, self.K, dtype=self.dtype,
                                device=self.variables["N"].device, keep=self.keep).mean()

    def __call__(self, order_params: dict) -> torch.Tensor:
        trigger_loss = (self.clusters["probability"].to(self.variables["N"].device)
                        * self.cluster_cross_entropy(order_params)).sum() / len(self.mus)
        q = self.trigger_probability
        return (1 - q) * self.non_trigger_loss(order_params) + q * trigger_loss

    def value_and_grad(self, order_params: dict) -> tuple[float, dict]:
        """The loss and its gradient with respect to every registered order
        parameter (missing ones are evaluated at 0), to the variances that are
        given, and to the profile if "M_profile" is given (a tensor; M_on then
        does not enter the logits and its gradient is 0)."""
        params = {name: torch.tensor(float(order_params.get(name, 0.0)), dtype=self.dtype, requires_grad=True)
                  for name in (*ORDER_PARAMS, *(name for name in VARIANCE_KEYS if name in order_params))}
        if PROFILE in order_params:
            params[PROFILE] = torch.as_tensor(order_params[PROFILE], dtype=self.dtype).detach().clone().requires_grad_(True)
        value = self(params)
        gradients = torch.autograd.grad(value, list(params.values()), allow_unused=True)
        return value.item(), {name: (torch.zeros_like(params[name]) if gradient is None else gradient).detach()
                              if params[name].dim() else (0.0 if gradient is None else gradient.item())
                              for name, gradient in zip(params, gradients)}

    def breakdown(self, order_params: dict) -> dict:
        """The loss by part, comparable with the logged metrics of the same names
        (LossMetric() / positions="trigg" / "non_trigg"), and per cluster:
        {"loss", "loss_trigg", "loss_non_trigg", "clusters"}, "clusters" being
        a QueryTable with "mu", "ell", "probability" (P(ell | mu)) and
        "cross_entropy". For a given ell use `trigger_loss(clusters, ell=...)`."""
        with torch.no_grad():
            clusters = QueryTable(dict(self.clusters))
            clusters["cross_entropy"] = self.cluster_cross_entropy(order_params).cpu()
            return {"loss": self(order_params).item(), "loss_trigg": trigger_loss(clusters),
                    "loss_non_trigg": float(self.non_trigger_loss(order_params)), "clusters": clusters}


def closure_cross_entropy(logits: torch.Tensor, variances: dict, K: int) -> torch.Tensor:
    """(rows,) the cross-entropy of Gaussian logits with means `logits` (rows, V, the
    canonical layout) and the variances of `noise_variances` (one per row), in the
    closure of eq. CE_closure: the K triggers share the common part xi_c and sum their
    individual parts by the law of large numbers, K e^{h_T + xi_c + V_i/2}; the
    non-targets likewise, sum_b e^{h_b + V_b/2}; the target and xi_c, jointly Gaussian,
    by the trapezoid rule on a grid of standard normals (QUADRATURE_SPACING,
    QUADRATURE_RANGE). With no variance this is logsumexp(logits) - logits[target]."""
    blocks = logit_blocks(K, logits.size(1))
    nodes = torch.arange(-QUADRATURE_RANGE, QUADRATURE_RANGE + QUADRATURE_SPACING / 2, QUADRATURE_SPACING,
                         dtype=logits.dtype, device=logits.device)
    weights = torch.exp(-nodes ** 2 / 2)
    weights = weights / weights.sum()                                               # the standard normal's
    target_normal, common_normal = (grid.reshape(-1) for grid in torch.meshgrid(nodes, nodes, indexing="ij"))
    weight = torch.outer(weights, weights).reshape(-1)
    xi_target, xi_common = _target_and_common(*(variances[name][:, None] for name in
                                                ("target", "trigger_common", "target_trigger")),
                                              target_normal, common_normal)       # (rows, nodes)
    target = logits[:, blocks["target"]] + xi_target
    trigger_mean = logits[:, :1]                                                   # the same for the K triggers
    triggers = math.log(K) + trigger_mean + variances["trigger_individual"].clamp(min=0)[:, None] / 2 + xi_common
    non_targets = torch.logsumexp(logits[:, blocks["non_target"]] + variances["non_target"].clamp(min=0) / 2, dim=1,
                                  keepdim=True)
    terms = torch.stack(torch.broadcast_tensors(target, triggers, non_targets), dim=-1)
    return ((torch.logsumexp(terms, dim=-1) - target) * weight).sum(1)


def trigger_loss(clusters: QueryTable, ell=None) -> float:
    """The mean cross-entropy over the trigger queries whose ell matches `ell`
    (None, k, [k, ...] or ">=k"), from a per-cluster table with "probability"
    and "cross_entropy" (`EffectiveLoss.breakdown(...)["clusters"]`): what
    LossMetric(positions="trigg", ell=ell) measures on a batch."""
    selected = condition_mask(clusters["ell"], ell)
    weights = clusters["probability"][selected]
    return ((weights * clusters["cross_entropy"][selected]).sum() / weights.sum()).item()


def asymptotic_loss(V: int, L: int, K: int) -> float:
    """The manuscript's L^infty: the loss of a perfect induction head, the floor
    log V on the positions it cannot solve, with lambda = L / (V + K) and
    rho = K / V,

        L^infty = (1 - omega_bar) log V + rho/(1+rho) log(1-rho) (1 - e^-lambda) / lambda,
        omega_bar = rho/(1+rho) (lambda - 1 + e^-lambda) / lambda.

    "How far training got" is then (log V - loss) / (log V - L^infty)."""
    rho, lam = K / V, L / (V + K)
    trigger = rho / (1 + rho)
    omega_bar = trigger * (lam - 1 + math.exp(-lam)) / lam
    return (1 - omega_bar) * math.log(V) + trigger * math.log(1 - rho) * (1 - math.exp(-lam)) / lam


def learning_time(loss: pd.Series, V: int, L: int, K: int, fraction: float = 0.2) -> float:
    """The first step at which `loss` (indexed by step: a run's logged loss or an
    `integrate` column) has gone `fraction` of the way from log V to
    `asymptotic_loss`, interpolated linearly between steps; NaN if never."""
    progress = (math.log(V) - loss.to_numpy(dtype=float)) / (math.log(V) - asymptotic_loss(V, L, K))
    reached = np.flatnonzero(progress >= fraction)
    if not len(reached):
        return math.nan
    steps, first = loss.index.to_numpy(dtype=float), reached[0]
    if first == 0:
        return steps[0]
    before, after = progress[first - 1], progress[first]
    return steps[first - 1] + (fraction - before) / (after - before) * (steps[first] - steps[first - 1])


def exact_rates(config, names) -> dict:
    """The `integrate` rates that make it the d = inf SGD dynamics of the ansatz (eqs.
    flow_mean, flow of the variance note), for a run's TrainerArgs (`RunData.config`).
    Every entry moves as d X_ij / d step = -eta_0 kappa_X d loss / d X_ij, with
    eta_0 = alpha_lr (the step of ReducedSGD), kappa_M = kappa_Gamma = 1 and
    kappa_Q = sigma_0^2 (the Gram matrix of the frozen WOV1), so the rate of

        a mean              eta_0 kappa_X / n      (n its support size)
        "M_profile"         eta_0                  (every entry; or one profile_family rate)
        a block variance    4 eta_0 kappa_X / n    (`integrate` multiplies it by the variance)
        a pooled variance   4 eta_0 kappa_X / n_X  (n_X the entries of all blocks of the matrix)

    At finite d the Gram matrices fluctuate around these values. Plain SGD only:
    momentum would rescale time."""
    if config.optim_args.momentum:
        raise ValueError("exact_rates is for plain SGD, and the run used momentum")
    L, V, K = config.model_args.seq_len, config.model_args.vocab_size, config.data_args.K
    eta_0 = config.optim_args.alpha_lr
    kappa = {"M": 1.0, "Q": config.model_args.sigma_0 ** 2, "G": 1.0}
    sizes = {**support_sizes(L, V, K), **variance_sizes(L, V, K)}
    matrix_of = {**{name: matrix for name, (matrix, _) in (*ORDER_PARAMS.items(), *VARIANCES.items())},
                 **POOLED_VARIANCES}
    rates = {}
    for name in names:
        if name == PROFILE:
            rates[name] = eta_0 * kappa["M"]
        elif name in matrix_of:
            rates[name] = (4 if name in VARIANCE_KEYS else 1) * eta_0 * kappa[matrix_of[name]] / sizes[name]
        else:
            raise ValueError(f"no rate for {name!r}: not a registered order parameter, profile or variance")
    return rates


def integrate(loss, initial_order_params: dict, rates: dict, steps: float, record_steps=None,
              method: str = "LSODA", rtol: float = 1e-6, atol: float = 1e-9, profile_family=None,
              **solver_options) -> pd.DataFrame:
    """The gradient flow of `loss` in the order parameters, in units of SGD steps:

        d theta_i / d step = - rates[i] * d loss / d theta_i     for every name in `rates`,

    while every other registered order parameter keeps its initial value (a
    missing one is 0). So freezing a parameter = leaving it out of `rates`; the
    M_on, Q_on, G_on-only model = only those three in `rates` and the others 0
    (or absent) in `initial_order_params`.

    loss: any callable {name: 0-d tensor} -> 0-d tensor, e.g. an EffectiveLoss.
    rates: {name: rate per SGD step}, derived by the caller (`exact_rates` for a run).

    The block variances (VARIANCES, POOLED_VARIANCES) flow multiplicatively,

        d var / d step = - rates[var] * var * d loss / d var,

    so a variance at 0 stays 0; one in `initial_order_params` but not in `rates` is
    passed to the loss unchanged. Each variance given is a column of the result.

    The profile can move too: put "M_profile" (a tensor of length L-1) in
    `initial_order_params` and in `rates`, with one rate for every entry or a
    tensor of rates. Its trajectory is the column "M_profile" (one numpy array
    per row), and "M_on" then reports the mean of the profile.

    Or the profile can move inside a family: profile_family = (function, theta0),
    with function(theta) -> the profile (a differentiable torch function of a
    (p,) tensor, returning length L-1) and theta0 the initial parameters; then
    "M_profile" goes in `rates` (one scalar rate per entry, e.g. alpha_lr at
    d = inf) but not in `initial_order_params`. theta follows the projection of
    the entry-wise flow d m / d step = - rate * d loss / d m on the family
    (least squares), with J = d profile / d theta of full rank:

        d theta / d step = - rate * (J^T J)^{-1} d loss / d theta.

    For the constant family this is the M_on flow with rate / (L-1). theta is
    the column "M_profile_params".
    steps: the flow runs over [0, steps]; record_steps: where to report it
    (default 201 evenly spaced points; e.g. `run.metrics.index` to overlay a run).
    method, rtol, atol, solver_options: passed to scipy.integrate.solve_ivp
    (LSODA switches to a stiff solver by itself).

    Returns a DataFrame indexed by "step" with one column per registered order
    parameter (the names of the logged metrics) and "loss". A note is warned and
    the rows reached so far are returned if the solver stops early. Note that
    all order parameters at 0 is a fixed point: start from non-zero values,
    e.g. `run.order_params("first")`.
    """
    if not rates:
        raise ValueError("rates is empty: name at least one order parameter to move")
    unknown = set(rates) - set(ORDER_PARAMS) - {PROFILE} - set(VARIANCE_KEYS)
    if unknown:
        raise ValueError(f"rates for unregistered order parameters: {sorted(unknown)}")
    names = (list(ORDER_PARAMS) + ([PROFILE] if PROFILE in initial_order_params or PROFILE in rates else [])
             + [name for name in VARIANCE_KEYS if name in initial_order_params or name in rates])
    profile_function = None
    if profile_family is not None:
        profile_function, theta0 = profile_family
        if PROFILE not in rates or PROFILE in initial_order_params or np.size(rates[PROFILE]) != 1:
            raise ValueError("with a profile_family, give M_profile one scalar rate and no initial value "
                             "(the family's theta0 is the start)")
        initial_order_params = dict(initial_order_params)
        initial_order_params[PROFILE] = torch.as_tensor(theta0, dtype=torch.float64)
    if PROFILE in rates and PROFILE not in initial_order_params:
        raise ValueError("a moving M_profile needs its initial value in initial_order_params")

    def initial(name) -> np.ndarray:
        value = initial_order_params.get(name, 0.0)
        return np.asarray(value.detach().cpu() if torch.is_tensor(value) else value, dtype=float).reshape(-1)

    moving = [name for name in names if name in rates]
    fixed = {name: (torch.as_tensor(initial(name)) if name == PROFILE else float(initial(name)[0]))
             for name in names if name not in rates}
    sizes = [initial(name).size for name in moving]
    rate_vector = np.concatenate([np.broadcast_to(np.asarray(rates[name], dtype=float).reshape(-1), (size,))
                                  for name, size in zip(moving, sizes)])
    is_variance = np.concatenate([np.full(size, name in VARIANCE_KEYS) for name, size in zip(moving, sizes)])

    def order_params_from(values) -> dict:
        params = dict(fixed)
        for name, piece in zip(moving, torch.split(torch.as_tensor(values), sizes)):
            if name != PROFILE:
                params[name] = piece[0]
            else:
                params[name] = piece if profile_function is None else profile_function(piece)
        return params

    if profile_function is not None:                    # where theta sits in the flat vector
        offset = sum(sizes[: moving.index(PROFILE)])
        theta_slice = slice(offset, offset + sizes[moving.index(PROFILE)])

    def velocity(step, values):
        flat = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        (gradient,) = torch.autograd.grad(loss(order_params_from(flat)), flat)
        if profile_function is not None:                # project the entry-wise flow on the family
            jacobian = torch.autograd.functional.jacobian(profile_function, flat.detach()[theta_slice])
            gradient[theta_slice] = torch.linalg.solve(jacobian.T @ jacobian, gradient[theta_slice])
        return -rate_vector * gradient.numpy() * np.where(is_variance, values, 1.0)

    record_steps = np.linspace(0, steps, 201) if record_steps is None else np.asarray(record_steps, dtype=float)
    start = np.concatenate([initial(name) for name in moving])
    solution = solve_ivp(velocity, (0.0, float(steps)), start, method=method, t_eval=record_steps,
                         rtol=rtol, atol=atol, **solver_options)
    if not solution.success:
        warnings.warn(f"the solver stopped at step {solution.t[-1] if len(solution.t) else 0}: {solution.message}")

    rows = []
    with torch.no_grad():
        for step, values in zip(solution.t, solution.y.T):
            params = order_params_from(values)
            row = {"step": step, "loss": float(loss(params))}
            for name, value in params.items():
                row[name] = value.numpy().copy() if name == PROFILE else float(value)
            if PROFILE in params:
                row["M_on"] = float(params[PROFILE].mean())
            if profile_function is not None:
                row["M_profile_params"] = values[theta_slice].copy()
            rows.append(row)
    extra = ["M_profile_params"] if profile_function is not None else []
    return pd.DataFrame(rows, columns=["step", *names, *extra, "loss"]).set_index("step")
