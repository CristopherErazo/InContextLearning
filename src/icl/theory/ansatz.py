"""The ansatz: which order parameters exist and how they make the logits.

This file and `variables.py` are the only places that know the ansatz
(paper/scratch/extended_ansatz.tex). To add an order parameter:

1. register it in ORDER_PARAMS: the matrix it lives on and its support;
2. add its terms to `ansatz_logits` (and, if they need a new sequence variable,
   add that variable to `variables.py`).

`measure_order_params`, `ansatz_matrices`, `support_sizes`, the probes, the
effective loss and the flow then pick it up without changes, and
tests/test_theory.py checks the new formula against the model itself.

Names follow the paper: M_on, Q_T, ... are the order parameters, and N, F, R,
W, P, U_bar, W_bar, P_bar the sequence variables of its Table 1.
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


def support_sizes(L: int, V: int, K: int) -> dict[str, int]:
    """Number of matrix entries each order parameter averages over."""
    return {name: int(support(L, V, K).sum()) for name, (_, support) in ORDER_PARAMS.items()}


def measure_order_params(matrices: dict, K: int) -> dict[str, float]:
    """Every registered order parameter of the matrices {"M", "Q", "G"} (the
    `model.matrices()` convention): the mean of its matrix over its support."""
    L, V = matrices["M"].shape[0], matrices["Q"].shape[0]
    order_params = {}
    for name, (matrix_name, support) in ORDER_PARAMS.items():
        matrix = torch.as_tensor(matrices[matrix_name])
        order_params[name] = matrix[support(L, V, K).to(matrix.device)].mean().item()
    return order_params


def ansatz_matrices(order_params: dict, L: int, V: int, K: int, dtype=torch.float64) -> dict[str, torch.Tensor]:
    """The matrices {"M", "Q", "G"} of the ansatz: each registered order
    parameter's value on its support, zero everywhere else (a parameter missing
    from `order_params` counts as 0). `measure_order_params` of the result gives
    `order_params` back."""
    shapes = {"M": (L, L), "Q": (V, V), "G": (V, V)}
    matrices = {name: torch.zeros(shape, dtype=dtype) for name, shape in shapes.items()}
    for name, (matrix_name, support) in ORDER_PARAMS.items():
        matrices[matrix_name][support(L, V, K)] = float(order_params.get(name, 0.0))
    return matrices


def ansatz_logits(variables: QueryTable, order_params: dict, beta: float, L: int) -> torch.Tensor:
    """(num_rows, V) logits of the ansatz in the canonical layout (eq.
    logit_classes_explicit), from the sequence variables (measured or sampled,
    see variables.py) and the order parameters ({name: float or 0-d tensor}; a
    missing one counts as 0). Differentiable in the order parameters.

        h_on  = beta/L G_on [M_on Q_on N + M_on Q_T F + M_off Q_T W + M_off dQ P]
        h_T   = beta/L G_T  [M_on Q_T n_keys + M_off Q_T sum_keys + M_on dQ N + M_off dQ R]
        h_b   = beta/L G_on [M_on Q_T U_bar_b + M_off Q_T W_bar_b + M_off dQ P_bar_b]

    with dQ = Q_on - Q_T, n_keys = max(mu-2, 0) the number of keys and
    sum_keys = n_keys (n_keys-1) / 2 the sum over the keys of (nu - 2).
    """
    dtype = variables["N"].dtype
    params = {name: torch.as_tensor(order_params.get(name, 0.0), dtype=dtype) for name in ORDER_PARAMS}
    M_on, M_off, Q_on, Q_T, G_on, G_T = (params[name] for name in ("M_on", "M_off", "Q_on", "Q_T", "G_on", "G_T"))
    delta_Q = Q_on - Q_T
    prefactor = beta / L
    n_keys = (variables["mu"] - 2).clamp(min=0).to(dtype)
    sum_keys = n_keys * (n_keys - 1) / 2

    N, F, R, W, P = (variables[name] for name in ("N", "F", "R", "W", "P"))
    h_target = prefactor * G_on * (M_on * Q_on * N + M_on * Q_T * F + M_off * Q_T * W + M_off * delta_Q * P)
    h_trigger = prefactor * G_T * (M_on * Q_T * n_keys + M_off * Q_T * sum_keys
                                   + M_on * delta_Q * N + M_off * delta_Q * R)
    h_non_target = prefactor * G_on * (M_on * Q_T * variables["U_bar"] + M_off * Q_T * variables["W_bar"]
                                       + M_off * delta_Q * variables["P_bar"])
    K = int(variables["K"])
    return torch.cat([h_trigger[:, None].expand(-1, K), h_non_target, h_target[:, None]], dim=1)
