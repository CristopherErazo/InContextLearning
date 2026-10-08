"""The deterministic sums over the keys of a query that every noise formula uses: the
order parameters as tensors, the mean previous-token entry of every key, the sums D, G1, K1,
K2, J2 of the mean M (eq. sums_bg, exact on the discrete positions) and the witness sums X1,
X2 (eq. X12). Shared by `icl.theory.terms` and re-exported by `noise`."""
from __future__ import annotations

import torch

from .ansatz import ORDER_PARAMS, PROFILE, VARIANCES, _check_profile, _take, block_variance


def _parameters(order_params: dict, dtype, device) -> dict[str, torch.Tensor]:
    """The means and the block variances (own key, else pooled, else 0) as tensors."""
    values = {name: order_params.get(name, 0.0) for name in ORDER_PARAMS}
    values |= {name: block_variance(order_params, name) for name in VARIANCES}
    return {name: torch.as_tensor(value, dtype=dtype, device=device) for name, value in values.items()}


def _diagonal(order_params: dict, L: int, dtype, device) -> torch.Tensor:
    """(L,) the mean previous-token entry M[j, j-1] of the key at code position j
    (0 at j = 0): the profile, else M_on."""
    if PROFILE in order_params:
        profile = _check_profile(order_params[PROFILE], L).to(dtype=dtype, device=device)
    else:
        profile = torch.as_tensor(order_params.get("M_on", 0.0), dtype=dtype, device=device).expand(L - 1)
    return torch.cat([profile.new_zeros(1), profile])


def _key_sums(diagonal: torch.Tensor, M_off, n_keys: torch.Tensor, var_M_on, var_M_off) -> dict:
    """The deterministic sums of eq. det_sums over the n keys of each row (code
    positions 1..n; key j has m_j on its previous token and M_off on its j - 1 other
    sources): D, G1, K1, K2, J2, and the diagonal summed ("all") and averaged ("mean")
    over the keys."""
    position = torch.arange(len(diagonal), dtype=diagonal.dtype, device=diagonal.device)
    m1, m2, position_m = (_take(values.cumsum(0), n_keys) for values in (diagonal, diagonal ** 2, position * diagonal))
    n = n_keys.to(diagonal.dtype)
    pairs = n * (n - 1) / 2                          # sum_j (j - 1)
    squares = (n - 1) * n * (2 * n - 1) / 6          # sum_j (j - 1)^2 = sum_j (n - j)^2
    return {"D": n * var_M_on + pairs * var_M_off,
            "G1": m2 + M_off ** 2 * pairs,
            "K1": m1 + M_off * pairs,
            "K2": m2 + 2 * M_off * (position_m - m1) + M_off ** 2 * squares,
            "J2": m2 + 2 * M_off * (n * m1 - position_m) + M_off ** 2 * squares,
            "all": m1, "mean": m1 / n.clamp(min=1)}


def _witness_sums(witness_keys: torch.Tensor, diagonal: torch.Tensor, M_off, n_keys: torch.Tensor) -> dict:
    """X1 = sum_nu k_nu n_a(nu - 2) and X2 = sum_nu n_a(nu - 2)^2 (eq. X12), and over the
    witness keys j the previous-token entry squared, m_j^2, and times the mean weight its
    source (an a) sends to all keys, m_j w = m_j (m_j + M_off (n - j)); from the witness
    keys w_k (code positions, 0 = padding): the key at j has n_a(nu - 2) = #{k: w_k < j}."""
    is_real = witness_keys > 0
    position = torch.arange(len(diagonal), dtype=diagonal.dtype, device=diagonal.device)
    key_weight_up_to = diagonal.cumsum(0) + M_off * position * (position - 1) / 2      # sum of k_i over i <= j
    X1 = ((_take(key_weight_up_to, n_keys)[:, None] - _take(key_weight_up_to, witness_keys)) * is_real).sum(1)
    later = torch.maximum(witness_keys[:, :, None], witness_keys[:, None, :])
    both = is_real[:, :, None] & is_real[:, None, :]
    X2 = ((n_keys[:, None, None] - later) * both).sum((1, 2)).to(diagonal.dtype)
    on_witness = _take(diagonal, witness_keys)
    return {"X1": X1, "X2": X2, "witness_squared": (on_witness ** 2).sum(1),
            "witness_sent": (on_witness * (on_witness + M_off * (n_keys[:, None] - witness_keys) * is_real)).sum(1)}
