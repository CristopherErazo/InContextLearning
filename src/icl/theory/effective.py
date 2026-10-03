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

`integrate` runs the gradient flow of such a loss in the order parameters,
with the rates (the Jacobian of the projection, the step size, ...) given by
the caller.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from scipy.stats import poisson

from .ansatz import ORDER_PARAMS, PROFILE, ansatz_logits
from .query_table import QueryTable, cluster_cross_entropy, condition_mask
from .variables import COUNT_LAWS, mean_variables, sample_variables

METHODS = ("mean", "mc")


class EffectiveLoss:
    """The effective population loss for a task (V, L, K) and a model with
    inverse temperature beta.

        loss = EffectiveLoss.from_config(run.config, method="mc", mus=range(16, 257, 16))
        loss(order_params)                       # 0-d tensor, differentiable in the values
        loss.value_and_grad(order_params)        # (float, {name: d loss / d name})
        loss.breakdown(order_params)             # loss per part and per (mu, ell) cluster

    `order_params` is {name: float or 0-d tensor} over the names of ORDER_PARAMS;
    a missing one counts as 0. ell is truncated where the Poisson tail falls
    below `ell_tol` (the kept probabilities are renormalised). For method="mc"
    the cached variables take about (number of clusters) x num_samples x V
    numbers: thin out `mus` for long sequences.
    """

    def __init__(self, V: int, L: int, K: int, beta: float, method: str = "mean", mus=None,
                 num_samples: int = 256, counts: str = "multinomial", ell_tol: float = 1e-6,
                 seed: int = 0, dtype=torch.float64, device=None):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if counts not in COUNT_LAWS:
            raise ValueError(f"counts must be one of {COUNT_LAWS}, got {counts!r}")
        self.V, self.L, self.K, self.beta, self.method = V, L, K, beta, method
        self.dtype, self.device = dtype, device
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
            self.variables = sample_variables(self.clusters["mu"].repeat_interleave(num_samples),
                                              self.clusters["ell"].repeat_interleave(num_samples),
                                              V, K, counts=counts, generator=generator, dtype=dtype,
                                              device=device)

    @classmethod
    def from_config(cls, config, **options) -> "EffectiveLoss":
        """From a run's TrainerArgs (e.g. `RunData.config`): V, L, K and beta."""
        return cls(config.model_args.vocab_size, config.model_args.seq_len, config.data_args.K,
                   config.model_args.beta, **options)

    # ---- the loss -------------------------------------------------------------

    def cluster_cross_entropy(self, order_params: dict) -> torch.Tensor:
        """(number of clusters,) CE(mu, ell), in the order of `self.clusters`."""
        logits = ansatz_logits(self.variables, order_params, self.beta, self.L)
        table = QueryTable({"logits": logits, "mu": self.variables["mu"], "ell": self.variables["ell"]})
        return cluster_cross_entropy(table)["cross_entropy"]

    def __call__(self, order_params: dict) -> torch.Tensor:
        trigger_loss = (self.clusters["probability"].to(self.variables["N"].device)
                        * self.cluster_cross_entropy(order_params)).sum() / len(self.mus)
        q = self.trigger_probability
        return (1 - q) * math.log(self.V) + q * trigger_loss

    def value_and_grad(self, order_params: dict) -> tuple[float, dict]:
        """The loss and its gradient with respect to every registered order
        parameter (missing ones are evaluated at 0), and with respect to the
        profile if "M_profile" is given (a tensor; M_on then does not enter the
        logits and its gradient is 0)."""
        params = {name: torch.tensor(float(order_params.get(name, 0.0)), dtype=self.dtype, requires_grad=True)
                  for name in ORDER_PARAMS}
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
                    "loss_non_trigg": math.log(self.V), "clusters": clusters}


def trigger_loss(clusters: QueryTable, ell=None) -> float:
    """The mean cross-entropy over the trigger queries whose ell matches `ell`
    (None, k, [k, ...] or ">=k"), from a per-cluster table with "probability"
    and "cross_entropy" (`EffectiveLoss.breakdown(...)["clusters"]`): what
    LossMetric(positions="trigg", ell=ell) measures on a batch."""
    selected = condition_mask(clusters["ell"], ell)
    weights = clusters["probability"][selected]
    return ((weights * clusters["cross_entropy"][selected]).sum() / weights.sum()).item()


def integrate(loss, initial_order_params: dict, rates: dict, steps: float, record_steps=None,
              method: str = "LSODA", rtol: float = 1e-6, atol: float = 1e-9, **solver_options) -> pd.DataFrame:
    """The gradient flow of `loss` in the order parameters, in units of SGD steps:

        d theta_i / d step = - rates[i] * d loss / d theta_i     for every name in `rates`,

    while every other registered order parameter keeps its initial value (a
    missing one is 0). So freezing a parameter = leaving it out of `rates`; the
    M_on, Q_on, G_on-only model = only those three in `rates` and the others 0
    (or absent) in `initial_order_params`.

    loss: any callable {name: 0-d tensor} -> 0-d tensor, e.g. an EffectiveLoss.
    rates: {name: rate per SGD step}, derived by the caller.

    The profile can move too: put "M_profile" (a tensor of length L-1) in
    `initial_order_params` and in `rates`, with one rate for every entry or a
    tensor of rates. Its trajectory is the column "M_profile" (one numpy array
    per row), and "M_on" then reports the mean of the profile.
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
    unknown = set(rates) - set(ORDER_PARAMS) - {PROFILE}
    if unknown:
        raise ValueError(f"rates for unregistered order parameters: {sorted(unknown)}")
    names = list(ORDER_PARAMS) + ([PROFILE] if PROFILE in initial_order_params or PROFILE in rates else [])
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

    def order_params_from(values) -> dict:
        params = dict(fixed)
        for name, piece in zip(moving, torch.split(torch.as_tensor(values), sizes)):
            params[name] = piece if name == PROFILE else piece[0]
        return params

    def velocity(step, values):
        flat = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        (gradient,) = torch.autograd.grad(loss(order_params_from(flat)), flat)
        return -rate_vector * gradient.numpy()

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
            rows.append(row)
    return pd.DataFrame(rows, columns=["step", *names, "loss"]).set_index("step")
