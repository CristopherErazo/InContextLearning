"""The fluctuation of the logits in the variance ansatz
(paper/scratch/2026-10-05-0036_variance-profile-ansatz-explicit.tex, sections "Logits"
and "Effective loss"). With block variances (VARIANCES, or the pooled ones), the
logits are h = h_bar + xi: h_bar is `ansatz_logits`, and xi has zero mean given the
sequence and the variances of eq. noise_trigger:

    noise_variances(variables, order_params, beta, L)       # per trigger query
    sample_logits(variables, order_params, beta, L, generator) # h_bar + a Gaussian xi
    non_trigger_loss(mu, order_params, beta, L, V, K)        # L_N(mu), eq. LN

Every variance is (beta/L)^2 times sums over the keys: the deterministic sums of
the mean previous-token matrix (D, G_1, K_1, K_2, J_2, exact on the discrete
positions), the sequence variables (measured or sampled: the key columns are
needed) and the coincidence rates kappa_2 = (V + 3K) p_T^2 and
kappa_bi = (K + 1 + 2K/V) p_T^2. A missing variance counts as 0: without any, every
variance is exactly 0, `sample_logits` is `ansatz_logits` and `non_trigger_loss` is
log V.
"""
from __future__ import annotations

import math

import torch

from .ansatz import (ORDER_PARAMS, PROFILE, VARIANCES, _brackets, _check_profile, _previous_token_sums, _take,
                     ansatz_logits, block_variance)
from .query_table import QueryTable, logit_blocks


def _parameters(order_params: dict, dtype, device) -> dict[str, torch.Tensor]:
    """The means and the block variances (own key, else pooled, else 0) as tensors."""
    values = {name: order_params.get(name, 0.0) for name in ORDER_PARAMS}
    values |= {name: block_variance(order_params, name) for name in VARIANCES}
    return {name: torch.as_tensor(value, dtype=dtype, device=device) for name, value in values.items()}


def _coincidence_rates(V: int, K: int) -> tuple[float, float]:
    """kappa_2, two keys hold the same token, and kappa_bi, the same token after the
    same predecessor (eqs. kappa2, kbi)."""
    p_T = 1 / (V + K)
    return (V + 3 * K) * p_T ** 2, (K + 1 + 2 * K / V) * p_T ** 2


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


def _background(counts: torch.Tensor, C_d, C_o, n) -> torch.Tensor:
    """Sigma^alpha(c) (eq. Sigma_alpha): the background attention noise on c keys at
    uniform positions, for counts (num_rows,) or (num_rows, tokens)."""
    if counts.dim() == 2:
        C_d, C_o, n = C_d[:, None], C_o[:, None], n[:, None]
    return counts * C_d / n + counts * (counts - 1) * C_o / n ** 2


def noise_variances(variables: QueryTable, order_params: dict, beta: float, L: int) -> dict[str, torch.Tensor]:
    """The variances of the logit fluctuation xi at the trigger queries of `variables`
    (measured or sampled: the key columns are needed), eq. noise_trigger:

        "target"              V_on (num_rows,)
        "trigger_common"      V_c^T, the part shared by the K triggers
        "trigger_individual"  V_i^T, i.i.d. over the triggers
        "non_target"          V_b^off (num_rows, V-K-1), in the `non_target` block order
        "target_trigger"      C^{on,T}, the covariance of the target and the common part

    Differentiable in the order parameters, the profile and the variances. Their mean
    given (mu, ell) is the average over `sample_variables` rows: they are not linear in
    the variables, so `mean_variables` would not give it."""
    dtype, device = variables["N"].dtype, variables["N"].device
    K = int(variables["K"])
    V = K + 1 + variables["U_bar"].size(1)
    kappa_2, kappa_bi = _coincidence_rates(V, K)
    p = _parameters(order_params, dtype, device)
    M_off, Q_T, G_on, G_T = (p[name] for name in ("M_off", "Q_T", "G_on", "G_T"))
    delta_Q = p["Q_on"] - Q_T
    q2_T, q2_on = Q_T ** 2 + p["var_Q_T"], p["Q_on"] ** 2 + p["var_Q_on"]
    g2_on = G_on ** 2 + p["var_G_on"]

    n_keys = (variables["mu"] - 2).clamp(min=0)
    n = n_keys.to(dtype).clamp(min=1)
    diagonal = _diagonal(order_params, L, dtype, device)
    sums = _key_sums(diagonal, M_off, n_keys, p["var_M_on"], p["var_M_off"])
    witness = _witness_sums(variables["N_keys"], diagonal, M_off, n_keys)
    on = _previous_token_sums(variables, order_params, L, dtype, device)
    brackets = _brackets(variables, order_params, L)
    B_phi, B_T, B_b = brackets["target"], brackets["trigger"], brackets["non_target"]
    ell, target_keys, P, R, P_bar = (variables["N"], variables["N"] + variables["F"], variables["P"], variables["R"],
                                     variables["P_bar"])

    # the mean attention on the a-sources over the target keys and over all keys (eq. Upsilon), and the
    # incoherent noise of the pairs whose source is an a (eq. AX)
    upsilon_P, upsilon_R = on["witness"] + M_off * P, on["witness"] + M_off * R
    a_pairs = lambda X: (q2_on - q2_T) * (p["var_M_on"] * ell + p["var_M_off"] * X)
    # the background on all keys, diagonal and off-diagonal (eq. background)
    C_d = q2_T * sums["D"] + p["var_Q_T"] * (sums["G1"] + kappa_2 * sums["K2"])
    C_o = p["var_Q_T"] * (sums["J2"] - sums["G1"] + kappa_2 * (sums["K1"] ** 2 - sums["K2"]))
    # the attention noise summed over all keys, the target keys, the keys of b (eqs. Call, Cphi, Cb)
    C_all = (q2_T * sums["D"] + p["var_Q_T"] * (sums["J2"] + kappa_2 * (sums["K1"] - upsilon_R) ** 2)
             + a_pairs(R) + p["var_Q_on"] * upsilon_R ** 2)
    C_phi = (_background(target_keys, C_d, C_o, n) - p["var_Q_T"] * witness["witness_squared"] + a_pairs(P)
             + p["var_Q_on"] * upsilon_P ** 2)
    output_pairs = variables["Y_bar"] * sums["mean"][:, None] ** 2           # (M_on)^2 Y_b, the profile at its mean
    C_b = (_background(variables["U_bar"], C_d, C_o, n) + p["var_Q_T"] * output_pairs
           + (q2_on - q2_T) * p["var_M_off"] * P_bar + p["var_Q_on"] * (M_off * P_bar) ** 2)
    # the readout energy of all keys (eq. E)
    energy = (B_phi ** 2 + p["var_Q_on"] * upsilon_P ** 2 + Q_T ** 2 * sums["K2"]
              + 2 * Q_T * delta_Q * M_off * witness["X1"] + (delta_Q * M_off) ** 2 * witness["X2"]
              + kappa_2 * (B_T - B_phi) ** 2 + C_d + kappa_2 * C_o + a_pairs(R)
              + p["var_Q_T"] * (kappa_bi - kappa_2 ** 2) * sums["all"] ** 2)

    # the row sums of C^{on,T}: on a witness key the previous-token source is an a, already in the var_Q_on term, so
    # what the background counts for it (its weight to all keys and its coincidences) comes out (missing in the note)
    witness_in_background = p["var_Q_T"] * (witness["witness_sent"] + kappa_2 * sums["K1"] * on["witness"])

    prefactor = (beta / L) ** 2
    return {
        "target": prefactor * (p["var_G_on"] * B_phi ** 2 + g2_on * C_phi + p["var_G_N"] * (energy - B_phi ** 2 - C_phi)),
        "trigger_common": prefactor * G_T ** 2 * C_all,
        "trigger_individual": prefactor * p["var_G_T"] * energy,
        "non_target": prefactor * (p["var_G_on"] * B_b ** 2 + g2_on * C_b
                                   + p["var_G_N"] * (energy[:, None] - B_b ** 2 - C_b)),
        "target_trigger": prefactor * G_on * G_T * (target_keys / n * (C_d + C_o) + a_pairs(P)
                                                    + p["var_Q_on"] * upsilon_P * upsilon_R - witness_in_background),
    }


def _sqrt(variance: torch.Tensor) -> torch.Tensor:
    """The standard deviation, 0 (with a zero gradient) where the variance is not positive."""
    positive = variance > 0
    return torch.where(positive, variance, 1).sqrt() * positive


def _target_and_common(target, common, covariance, target_normal, common_normal) -> tuple[torch.Tensor, torch.Tensor]:
    """(xi_on, xi_c^T) from two standard normals, through the Cholesky factor of their
    covariance [[target, covariance], [covariance, common]] (broadcast); what is not
    positive counts as 0."""
    sd_target = _sqrt(target)
    has_target = sd_target > 0
    loading = covariance / torch.where(has_target, sd_target, 1) * has_target
    return sd_target * target_normal, loading * target_normal + _sqrt(common - loading ** 2) * common_normal


def sample_logits(variables: QueryTable, order_params: dict, beta: float, L: int,
                  generator: torch.Generator | None = None, normals: torch.Tensor | None = None) -> torch.Tensor:
    """(num_rows, V) logits in the canonical layout: `ansatz_logits` plus a Gaussian draw
    of the fluctuation with the variances of `noise_variances`, the Gaussian version of
    the note's sampler: the target and the common trigger part jointly (with their
    covariance), the individual trigger parts i.i.d., the non-targets independent.

    `normals`: (num_rows, V + 1) standard normals, one per entry of the canonical layout
    and the common trigger part last; fixed normals are common random numbers (the
    result is then smooth in the order parameters). Drawn with `generator` if not given.
    The formulas are leading order: a negative variance counts as 0."""
    logits = ansatz_logits(variables, order_params, beta, L)
    variances = noise_variances(variables, order_params, beta, L)
    num_rows, V = logits.shape
    blocks = logit_blocks(int(variables["K"]), V)
    if normals is None:
        normals = torch.randn(num_rows, V + 1, generator=generator, dtype=logits.dtype, device=logits.device)
    target, common = _target_and_common(variances["target"], variances["trigger_common"], variances["target_trigger"],
                                        normals[:, blocks["target"]].squeeze(1), normals[:, -1])
    triggers = common[:, None] + _sqrt(variances["trigger_individual"])[:, None] * normals[:, blocks["triggers"]]
    non_target = _sqrt(variances["non_target"]) * normals[:, blocks["non_target"]]
    return logits + torch.cat([triggers, non_target, target[:, None]], dim=1)


def non_trigger_loss(mu, order_params: dict, beta: float, L: int, V: int, K: int,
                     dtype=torch.float64, device=None) -> torch.Tensor:
    """The expected cross-entropy at a non-trigger query at position mu (int or 1-D),
    to second order in the fluctuation (eq. LN): log V plus half the variance of the
    logit vector about its own mean, the common modes of the trigger and non-trigger
    entries included. Exactly log V without variances."""
    mu = torch.as_tensor(mu, device=device).reshape(-1)
    kappa_2, kappa_bi = _coincidence_rates(V, K)
    p_T, theta = 1 / (V + K), K / V
    p = _parameters(order_params, dtype, device)
    G_on, G_T, var_Q = p["G_on"], p["G_T"], p["var_Q_N"]
    sums = _key_sums(_diagonal(order_params, L, dtype, device), p["M_off"], (mu - 2).clamp(min=0),
                     p["var_M_on"], p["var_M_off"])
    # the noise of a query row of Q with zero mean (eq. noise_nontrigger)
    C_d = var_Q * (sums["D"] + sums["G1"] + kappa_2 * sums["K2"])
    C_o = var_Q * (sums["J2"] - sums["G1"] + kappa_2 * (sums["K1"] ** 2 - sums["K2"]))
    energy = C_d + kappa_2 * C_o + var_Q * (kappa_bi - kappa_2 ** 2) * sums["all"] ** 2
    # the weights of eq. omegas: own variances minus the common trigger mode and the trigger / non-trigger covariance
    readout = G_on ** 2 + p["var_G_on"] - p["var_G_N"]
    shared = theta * (1 - theta) * G_T ** 2 - 2 * theta * p_T * G_T * G_on
    omega_d = shared + p_T * (readout - G_on ** 2 / V)
    omega_o = shared + p_T ** 2 * ((1 + 2 * theta) * readout - G_on ** 2)
    omega_energy = (1 - 1 / V) * (theta * p["var_G_T"] + (1 - theta) * p["var_G_N"])
    output_pairs = readout * var_Q * theta * (p_T * mu.to(dtype) * sums["mean"]) ** 2
    return math.log(V) + 0.5 * (beta / L) ** 2 * (omega_d * C_d + omega_o * C_o + omega_energy * energy + output_pairs)
