"""The ansatz: which order parameters exist and how they make the logits.

This file and `variables.py` are the only places that know the ansatz
(paper/scratch/extended_ansatz.tex); `noise.py` adds the fluctuation of the
variance ansatz on top of the brackets of `ansatz_logits`. To add an order parameter:

1. register it in ORDER_PARAMS: the matrix it lives on and its support;
2. add its terms to `ansatz_logits` (and, if they need a new sequence variable,
   add that variable to `variables.py`).

`measure_order_params`, `ansatz_matrices`, `support_sizes`, the probes, the
effective loss and the flow then pick it up without changes, and
tests/test_theory.py checks the new formula against the model itself.

Names follow the paper: M_on, Q_T, ... are the order parameters, and N, F, R,
W, P, U_bar, W_bar, P_bar the sequence variables of its Table 1.

The previous-token diagonal can also be a *profile*
(paper/scratch/2026-10-03-1539_profile-ansatz.tex): give
`order_params["M_profile"]`, a tensor of length L-1 in the convention of
`M.diagonal(-1)` (entry i is M[i+1, i], the key at code position i+1). It
replaces M_on in the logits and in `ansatz_matrices`. M_on stays the scalar that
is measured and logged (the mean of the profile).

The variance ansatz (paper/scratch/2026-10-05-0036_variance-profile-ansatz-explicit.tex)
adds the spread of every block around its mean: VARIANCES holds the eight block
variances ("var_M_on", ..., "var_G_N"), POOLED_VARIANCES the one-per-matrix ones
("var_M", "var_Q", "var_G") that stand in for any block of the matrix without its own
key. Like the means, a missing variance counts as 0, so no variance key = no noise.
LEVELS names the keys each simplification level of the note keeps (`at_level`).
"""
from __future__ import annotations

import torch

from .query_table import QueryTable


def _previous_token(L, V, K):          # M_{nu, nu-1}
    return torch.diag(torch.ones(L - 1), -1).bool()


def _bulk(L, V, K):                    # M_{nu, sigma}, sigma <= nu - 2
    return torch.ones(L, L).tril(-2).bool()


def _trigger_rows(V, K):
    return (torch.arange(V) < K)[:, None].expand(V, V)


def _trigger_diagonal(L, V, K):        # Q_{aa}, a trigger
    return torch.eye(V, dtype=torch.bool) & _trigger_rows(V, K)


def _trigger_off_diagonal(L, V, K):    # Q_{ac}, a trigger, c != a
    return _trigger_rows(V, K) & ~torch.eye(V, dtype=torch.bool)


def _non_trigger_diagonal(L, V, K):    # Gamma_{bb}, b not a trigger
    return torch.eye(V, dtype=torch.bool) & ~_trigger_rows(V, K)


def _all_trigger_rows(L, V, K):        # Gamma_{tau b}, tau a trigger, every b
    return _trigger_rows(V, K).clone()


def _non_trigger_rows(L, V, K):        # Q_{ac}, a not a trigger, every c
    return ~_trigger_rows(V, K)


def _non_trigger_off_diagonal(L, V, K):    # Gamma_{tau b}, tau not a trigger, b != tau
    return ~_trigger_rows(V, K) & ~torch.eye(V, dtype=torch.bool)


PROFILE = "M_profile"                  # optional key: the previous-token diagonal as a tensor

# name -> (matrix, support(L, V, K) -> bool mask of that matrix's shape).
# The value of an order parameter is the mean of its matrix over its support.
ORDER_PARAMS = {
    "M_on": ("M", _previous_token),
    "M_off": ("M", _bulk),
    "Q_on": ("Q", _trigger_diagonal),
    "Q_T": ("Q", _trigger_off_diagonal),
    "G_on": ("G", _non_trigger_diagonal),
    "G_T": ("G", _all_trigger_rows),
}

# Means that are 0 in the ansatz (no task-aligned direction), measured as diagnostics only.
DIAGNOSTIC_MEANS = {
    "Q_N": ("Q", _non_trigger_rows),
    "G_N": ("G", _non_trigger_off_diagonal),
}

# name -> (matrix, support): the eight blocks of the variance ansatz. The value is the
# variance of the block's entries around their own mean, except "var_M_on", the spread
# of the sub-diagonal around its smooth profile (see `measure_variances`).
VARIANCES = {
    "var_M_on": ("M", _previous_token),
    "var_M_off": ("M", _bulk),
    "var_Q_on": ("Q", _trigger_diagonal),
    "var_Q_T": ("Q", _trigger_off_diagonal),
    "var_Q_N": ("Q", _non_trigger_rows),
    "var_G_on": ("G", _non_trigger_diagonal),
    "var_G_T": ("G", _all_trigger_rows),
    "var_G_N": ("G", _non_trigger_off_diagonal),
}

# name -> matrix: one variance per matrix, the pooled within-block variance (the
# size-weighted mean of its blocks' variances, not the raw variance of the matrix).
POOLED_VARIANCES = {"var_M": "M", "var_Q": "Q", "var_G": "G"}

SIGNAL = ("M_on", "Q_on", "G_on")

# The simplification levels of the note: the keys each level keeps (the rest count as 0).
LEVELS = {
    "S0": (*ORDER_PARAMS, PROFILE, *VARIANCES),            # profile, six means, eight variances
    "S1": (*ORDER_PARAMS, PROFILE, *POOLED_VARIANCES),     # one variance per matrix
    "S2": (*ORDER_PARAMS, *POOLED_VARIANCES),              # no profile
    "S3": tuple(ORDER_PARAMS),                             # no spread: the extended ansatz
    "S4": (*SIGNAL, *POOLED_VARIANCES),                    # signal + noise floor
    "S5": SIGNAL,                                          # signal only (the progress report)
}


def at_level(order_params: dict, level: str) -> dict:
    """The entries of `order_params` that `level` (a key of LEVELS) keeps; the others
    are dropped, i.e. count as 0. E.g. `at_level(run.order_params(0, variances=True), "S5")`
    is the signal-only model."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {list(LEVELS)}, got {level!r}")
    return {name: value for name, value in order_params.items() if name in LEVELS[level]}


def block_variance(order_params: dict, name: str):
    """The variance of the block `name` (a key of VARIANCES): its own entry of
    `order_params`, else the pooled one of its matrix, else 0."""
    pooled = "var_" + VARIANCES[name][0]
    return order_params.get(name, order_params.get(pooled, 0.0))


def support_sizes(L: int, V: int, K: int) -> dict[str, int]:
    """Number of matrix entries each order parameter averages over."""
    return {name: int(support(L, V, K).sum()) for name, (_, support) in ORDER_PARAMS.items()}


def variance_sizes(L: int, V: int, K: int) -> dict[str, int]:
    """Number of entries of each block of VARIANCES, and of each matrix for the
    pooled ones of POOLED_VARIANCES (the sum of its blocks)."""
    sizes = {name: int(support(L, V, K).sum()) for name, (_, support) in VARIANCES.items()}
    pooled_sizes = {pooled: sum(size for name, size in sizes.items() if VARIANCES[name][0] == matrix_name)
                    for pooled, matrix_name in POOLED_VARIANCES.items()}
    return {**sizes, **pooled_sizes}


def _block_entries(matrices: dict, registry: dict, K: int) -> dict[str, torch.Tensor]:
    L, V = matrices["M"].shape[0], matrices["Q"].shape[0]
    entries = {}
    for name, (matrix_name, support) in registry.items():
        matrix = torch.as_tensor(matrices[matrix_name])
        entries[name] = matrix[support(L, V, K).to(matrix.device)]
    return entries


def measure_order_params(matrices: dict, K: int) -> dict[str, float]:
    """Every registered order parameter of the matrices {"M", "Q", "G"} (the
    `model.matrices()` convention): the mean of its matrix over its support."""
    return {name: entries.mean().item() for name, entries in _block_entries(matrices, ORDER_PARAMS, K).items()}


def measure_diagnostic_means(matrices: dict, K: int) -> dict[str, float]:
    """The means of DIAGNOSTIC_MEANS (Q_N, G_N), which the ansatz sets to 0."""
    return {name: entries.mean().item() for name, entries in _block_entries(matrices, DIAGNOSTIC_MEANS, K).items()}


def measure_variances(matrices: dict, K: int) -> dict[str, float]:
    """The block variances of the matrices {"M", "Q", "G"}: the eight of VARIANCES
    (each block's entries around their own mean), the three of POOLED_VARIANCES and
    "var_M_on_raw".

    "var_M_on" is the spread of the sub-diagonal around its smooth profile, half the
    mean squared difference of neighbouring entries (biased by (M_on phi')^2 / (2 L^2)
    only); "var_M_on_raw" is its spread around M_on, profile included: the value a
    model without the profile sees."""
    entries = {name: values.double() for name, values in _block_entries(matrices, VARIANCES, K).items()}
    variances = {name: values.var(unbiased=False).item() for name, values in entries.items()}
    sub_diagonal = torch.as_tensor(matrices["M"]).double().diagonal(-1)
    raw = variances["var_M_on"]
    variances["var_M_on"] = 0.5 * sub_diagonal.diff().pow(2).mean().item()
    for pooled, matrix_name in POOLED_VARIANCES.items():
        blocks = [name for name, (block_matrix, _) in VARIANCES.items() if block_matrix == matrix_name]
        size = sum(entries[name].numel() for name in blocks)
        variances[pooled] = sum(entries[name].numel() * variances[name] for name in blocks) / size
    variances["var_M_on_raw"] = raw
    return variances


def ansatz_matrices(order_params: dict, L: int, V: int, K: int, dtype=torch.float64,
                    noise: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """The matrices {"M", "Q", "G"} of the ansatz: each registered order
    parameter's value on its support, zero everywhere else (a parameter missing
    from `order_params` counts as 0), and the profile on the sub-diagonal of M if
    `order_params["M_profile"]` is given. `measure_order_params` of the result
    gives the scalar order parameters back (M_on as the mean of the profile).

    With a generator as `noise`, every block also gets i.i.d. Gaussian fluctuations
    with its variance in `order_params` (`block_variance`: own key, else pooled,
    else none): a draw of the variance ansatz."""
    shapes = {"M": (L, L), "Q": (V, V), "G": (V, V)}
    matrices = {name: torch.zeros(shape, dtype=dtype) for name, shape in shapes.items()}
    for name, (matrix_name, support) in ORDER_PARAMS.items():
        matrices[matrix_name][support(L, V, K)] = float(order_params.get(name, 0.0))
    if PROFILE in order_params:
        matrices["M"].diagonal(-1).copy_(_check_profile(order_params[PROFILE], L).detach())
    if noise is not None:
        for name, (matrix_name, support) in VARIANCES.items():
            variance = float(block_variance(order_params, name))
            if variance:
                mask = support(L, V, K)
                matrices[matrix_name][mask] += variance ** 0.5 * torch.randn(int(mask.sum()), generator=noise,
                                                                           dtype=dtype)
    return matrices


def _check_profile(profile, L: int) -> torch.Tensor:
    profile = torch.as_tensor(profile)
    if profile.shape != (L - 1,):
        raise ValueError(f"M_profile must have shape ({L - 1},) (the sub-diagonal of M), got {tuple(profile.shape)}")
    return profile


def _previous_token_sums(variables: QueryTable, order_params: dict, L: int, dtype, device) -> dict:
    """The previous-token diagonal summed over the keys each logit uses:
    "witness" (the keys after the earlier a's), "free" (the free target keys),
    "tokens" (each non-target token's keys, (n, V-K-1)) and "all" (every key).

    Without a profile these are M_on times the counts. With a profile they read
    it at the keys stored in the variables ("N_keys", "F_keys", "U_keys", code
    positions, 0 = padding); a table without keys (`mean_variables`) gets the
    count times the mean of the profile over the query's keys."""
    n_keys = (variables["mu"] - 2).clamp(min=0)
    if PROFILE not in order_params:
        M_on = torch.as_tensor(order_params.get("M_on", 0.0), dtype=dtype, device=device)
        return {"witness": M_on * variables["N"], "free": M_on * variables["F"],
                "tokens": M_on * variables["U_bar"], "all": M_on * n_keys.to(dtype)}
    profile = _check_profile(order_params[PROFILE], L).to(dtype=dtype, device=device)
    by_code_position = torch.cat([profile.new_zeros(1), profile])      # key at code position j -> M[j, j-1]
    total = by_code_position.cumsum(0)[n_keys]                            # keys are code positions 1..mu-2
    window_mean = total / n_keys.clamp(min=1).to(dtype)
    sums = {"all": total}
    for name, keys, count in (("witness", "N_keys", "N"), ("free", "F_keys", "F"), ("tokens", "U_keys", "U_bar")):
        if keys in variables:
            sums[name] = by_code_position[variables[keys]].sum(-1)
        else:
            mean = window_mean if variables[count].dim() == 1 else window_mean[:, None]
            sums[name] = variables[count] * mean
    return sums


def ansatz_logits(variables: QueryTable, order_params: dict, beta: float, L: int) -> torch.Tensor:
    """(num_rows, V) logits of the ansatz in the canonical layout (eq.
    logit_classes_explicit), from the sequence variables (measured or sampled,
    see variables.py) and the order parameters ({name: float or 0-d tensor},
    plus optionally "M_profile"; a missing one counts as 0). Differentiable in
    the order parameters, the profile included.

        h_on  = beta/L G_on [Q_on N~ + Q_T F~ + M_off Q_T W + M_off dQ P]
        h_T   = beta/L G_T  [Q_T M_all + M_off Q_T sum_keys + dQ N~ + M_off dQ R]
        h_b   = beta/L G_on [Q_T U~_b + M_off Q_T W_bar_b + M_off dQ P_bar_b]

    with dQ = Q_on - Q_T and sum_keys = n_keys (n_keys-1) / 2, n_keys = max(mu-2, 0).
    N~, F~, U~_b, M_all are the previous-token diagonal summed over the witness
    keys, the free target keys, the keys of b and all the keys: M_on N, M_on F,
    M_on U_bar_b, M_on n_keys without a profile (see `_previous_token_sums`).
    """
    dtype, device = variables["N"].dtype, variables["N"].device
    G_on, G_T = (torch.as_tensor(order_params.get(name, 0.0), dtype=dtype, device=device) for name in ("G_on", "G_T"))
    prefactor = beta / L
    brackets = _brackets(variables, order_params, L)
    h_target = prefactor * G_on * brackets["target"]
    h_trigger = prefactor * G_T * brackets["trigger"]
    h_non_target = prefactor * G_on * brackets["non_target"]
    K = int(variables["K"])
    return torch.cat([h_trigger[:, None].expand(-1, K), h_non_target, h_target[:, None]], dim=1)


def _brackets(variables: QueryTable, order_params: dict, L: int) -> dict[str, torch.Tensor]:
    """The brackets of `ansatz_logits`, h = beta/L Gamma B (eq. Bc of the variance note):
    "target" B_phi and "trigger" B_T (num_rows,), "non_target" B_b (num_rows, V-K-1)."""
    dtype, device = variables["N"].dtype, variables["N"].device
    M_off, Q_on, Q_T = (torch.as_tensor(order_params.get(name, 0.0), dtype=dtype, device=device)
                        for name in ("M_off", "Q_on", "Q_T"))
    delta_Q = Q_on - Q_T
    n_keys = (variables["mu"] - 2).clamp(min=0).to(dtype)
    sum_keys = n_keys * (n_keys - 1) / 2
    on = _previous_token_sums(variables, order_params, L, dtype, device)
    R, W, P = (variables[name] for name in ("R", "W", "P"))
    return {"target": Q_on * on["witness"] + Q_T * on["free"] + M_off * Q_T * W + M_off * delta_Q * P,
            "trigger": Q_T * on["all"] + M_off * Q_T * sum_keys + delta_Q * on["witness"] + M_off * delta_Q * R,
            "non_target": (Q_T * on["tokens"] + M_off * Q_T * variables["W_bar"]
                           + M_off * delta_Q * variables["P_bar"])}
