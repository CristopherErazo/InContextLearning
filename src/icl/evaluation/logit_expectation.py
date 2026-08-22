"""
Expected on-target logit for the 2-layer linear-attention transformer of
sec:transformer-definition, following the closed-form derivation of
sec:apx_logit-expectation-derivation.

Everything is vectorized in torch; no Python-level loops over V, L or K.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Step 0 — build Q, Gamma, M from the primitive model matrices
# --------------------------------------------------------------------------- #
@dataclass
class DerivedMatrices:
    M: Tensor      # (L, L)   = P^T W_QK1 P
    Q: Tensor      # (V, V)   = E^T W_QK2 W_OV1 E
    Gamma: Tensor  # (V, V)   = U W_OV2 E


def build_derived_matrices(
    E: Tensor, P: Tensor, U: Tensor,
    W_QK1: Tensor, W_OV1: Tensor, W_QK2: Tensor, W_OV2: Tensor,
) -> DerivedMatrices:
    """E: (d,V), P: (d,L), U: (V,d), W_*: (d,d)."""
    M = P.T @ W_QK1 @ P                      # (L, L)
    Q = E.T @ W_QK2 @ W_OV1 @ E               # (V, V)
    Gamma = U @ W_OV2 @ E                     # (V, V)
    return DerivedMatrices(M=M, Q=Q, Gamma=Gamma)


# --------------------------------------------------------------------------- #
# Step 1 — positional sums S0..S3(mu), from M and rho only
# --------------------------------------------------------------------------- #
def _c_n(n: Tensor, V: int, rho: float) -> Tensor:
    """c_n = (-1)^n rho^n / (V (1+rho) rho^2), n given as a (possibly signed) tensor.
    Entries with n <= 0 are masked to 0 by the caller (they never correspond
    to a valid gap)."""
    return ((-rho) ** n) / (V * (1 + rho) * rho ** 2)


def positional_sums(M: Tensor, V: int, rho: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Returns S0, S1, S2, S3 as (L,) tensors indexed by mu (0-indexed position).
    S_i(mu) = 0 automatically for mu < 2 (no valid sigma < nu < mu)."""
    L = M.shape[0]
    device, dtype = M.device, M.dtype
    idx = torch.arange(L, device=device)
    gap = idx[:, None] - idx[None, :]                 # gap[i,j] = i - j
    lower_strict = gap > 0                             # j < i

    c_gap = torch.zeros((L, L), device=device, dtype=dtype)
    c_gap[lower_strict] = _c_n(gap[lower_strict].to(dtype), V, rho)

    # cumM[nu]  = sum_{sigma<nu} M[nu,sigma]
    # cumMc[nu] = sum_{sigma<nu} M[nu,sigma] * c_{nu-sigma}
    tri_mask = lower_strict.to(dtype)                  # mask[nu,sigma] = 1{sigma<nu}
    cumM = (M * tri_mask).sum(dim=1)                   # (L,)
    cumMc = (M * tri_mask * c_gap).sum(dim=1)           # (L,)

    # S0(mu) = sum_{nu<mu} cumM[nu]  ;  S2(mu) = sum_{nu<mu} cumMc[nu]
    S0 = torch.cumsum(cumM, dim=0) - cumM               # exclusive prefix sum
    S2 = torch.cumsum(cumMc, dim=0) - cumMc

    # S1(mu) = sum_{nu<mu} c_{mu-nu} cumM[nu]  = (C1 @ cumM)[mu], C1[mu,nu]=c_{mu-nu} if nu<mu
    C1 = c_gap  # same lower-strict Toeplitz-like structure, reused
    S1 = C1 @ cumM
    S3 = C1 @ cumMc
    return S0, S1, S2, S3


def adjacent_position_correction_weight(M: Tensor, V: int, rho: float) -> Tensor:
    """K1(mu) = pi(a) * M1(mu) - rho^2 * M1c(mu), the scalar weight (independent
    of the trigger a, since pi(a) is constant over a in T) that multiplies the
    adjacent-position (n=1 kernel) correction. Returns a (L,) tensor indexed by mu.

    This exists because the closed-form kernel P^{(n)}_{xy}=pi(y)-c_n r(x)s(y)
    is only exact for n>=2; every (sigma,nu) pair with nu-sigma=1 needs this
    correction (pairs with mu-nu=1 need none, see note in the derivation)."""
    L = M.shape[0]
    device, dtype = M.device, M.dtype
    idx = torch.arange(L, device=device)
    gap = idx[:, None] - idx[None, :]
    lower_strict = gap > 0
    c_gap = torch.zeros((L, L), device=device, dtype=dtype)
    c_gap[lower_strict] = _c_n(gap[lower_strict].to(dtype), V, rho)

    v = torch.zeros(L, device=device, dtype=dtype)
    if L > 1:
        v[1:] = torch.diagonal(M, offset=-1)            # v[nu] = M[nu,nu-1], nu>=1
    M1 = torch.cumsum(v, dim=0) - v                       # exclusive prefix sum
    M1c = c_gap @ v

    pi_a = 1.0 / (V * (1 + rho))                           # constant for any a in T
    return pi_a * M1 - rho ** 2 * M1c


def adjacent_position_correction_term(
    Q: Tensor, Gamma: Tensor, trigger_idx: Tensor, V: int, rho: float,
) -> Tensor:
    """CorrTerm(a), the trigger-dependent (mu-independent) factor of the
    adjacent-position correction. Returns a (K,) tensor."""
    device, dtype = Q.device, Q.dtype
    K = trigger_idx.numel()
    all_idx = torch.arange(V, device=device)
    is_trig = torch.zeros(V, dtype=torch.bool, device=device)
    is_trig[trigger_idx] = True
    tau_idx = all_idx[~is_trig]

    D_T = _trigger_indicator(V, trigger_idx, device, dtype)

    Q_diag = Q.diagonal()
    Q_T_rowsum = (Q * D_T[None, :]).sum(dim=1)             # (V,) = Q_a^T

    Gamma_diag = Gamma.diagonal()
    Gamma_Tbar_row = (Gamma * (1.0 - D_T)[None, :]).sum(dim=1)  # (V,) = Gamma_tau^{Tbar}
    Gamma_Gtau = Gamma_Tbar_row - Gamma_diag               # (V,) = Gamma_tau^{G_tau}, valid for tau not in T

    Gamma_bar_diag = Gamma_diag[tau_idx].mean()
    Gamma_bar_Gtau = Gamma_Gtau[tau_idx].mean()

    N = V - K - 1
    n = K - 1
    Q_aa = Q_diag[trigger_idx]                              # (K,)
    Q_aT = Q_T_rowsum[trigger_idx]                           # (K,)

    corr = Gamma_bar_diag * (Q_aa - Q_aT / (V * rho))
    if N > 0:
        corr = corr + (Gamma_bar_Gtau / N) * (Q_aT - Q_aa - n * Q_aT / (V * rho))
    return corr


# --------------------------------------------------------------------------- #
# Step 2 — hypergeometric moments of (X, Y) given tau, for every (a, tau) pair
# --------------------------------------------------------------------------- #
def _trigger_indicator(V: int, trigger_idx: Tensor, device, dtype) -> Tensor:
    d = torch.zeros(V, device=device, dtype=dtype)
    d[trigger_idx] = 1.0
    return d


def _xy_moments(
    Q: Tensor, Gamma: Tensor, D_Tbar: Tensor,
    a_idx: Tensor, tau_idx: Tensor, K: int, V: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Returns Q_at, Gamma_tt, EX, EY, EXY  — all shape (K, N) except Gamma_tt (N,)."""
    n = K - 1
    Npop = V - K - 1
    Q_at = Q[a_idx][:, tau_idx]                                   # (K,N)
    Gamma_tt = Gamma.diagonal()[tau_idx]                          # (N,)

    Q_Tbar_row = (Q * D_Tbar[None, :]).sum(dim=1)                 # (V,) = Q_a^{Tbar}
    Gamma_Tbar_row = (Gamma * D_Tbar[None, :]).sum(dim=1)         # (V,) = Gamma_tau^{Tbar}

    Q_a_Gtau = Q_Tbar_row[a_idx][:, None] - Q_at                  # (K,N)
    Gamma_tau_Gtau = Gamma_Tbar_row[tau_idx][None, :] - Gamma_tt[None, :]  # (1,N)

    QD = Q * D_Tbar[None, :]                                      # (V,V), zero out trigger cols
    R = QD @ Gamma.T                                               # (V,V), R[a,tau] = R_a(tau)
    R_at = R[a_idx][:, tau_idx]                                    # (K,N)
    R_a_Gtau = R_at - Q_at * Gamma_tt[None, :]                     # (K,N)

    if Npop > 0:
        f1 = n / Npop
    else:
        f1 = 0.0
    if Npop > 1:
        f2 = (n * (n - 1)) / (Npop * (Npop - 1))
    else:
        f2 = 0.0

    EX = Q_at + f1 * Q_a_Gtau                                      # (K,N)
    EY = Gamma_tt[None, :] + f1 * Gamma_tau_Gtau                   # (1,N) -> broadcasts
    EXY = (
        Q_at * Gamma_tt[None, :]
        + f1 * (Q_at * Gamma_tau_Gtau + Gamma_tt[None, :] * Q_a_Gtau)
        + f1 * R_a_Gtau
        + f2 * (Q_a_Gtau * Gamma_tau_Gtau - R_a_Gtau)
    )
    return Q_at, Gamma_tt, EX, EY, EXY


def _bilinear_expectation(p0: Tensor, p1, q0: Tensor, q1, EX: Tensor, EY: Tensor, EXY: Tensor) -> Tensor:
    """E[(p0 + p1 X)(q0 + q1 Y)] = p0 q0 + p0 q1 EY + p1 q0 EX + p1 q1 EXY."""
    return p0 * q0 + p0 * q1 * EY + p1 * q0 * EX + p1 * q1 * EXY


# --------------------------------------------------------------------------- #
# Step 3 — bar{AC}, bar{AD}, bar{BE}, bar{BF} per trigger a  (averaged over tau)
# --------------------------------------------------------------------------- #
def _trigger_bar_terms(
    Q: Tensor, Gamma: Tensor, trigger_idx: Tensor, V: int, rho: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    device, dtype = Q.device, Q.dtype
    K = trigger_idx.numel()
    all_idx = torch.arange(V, device=device)
    is_trig = torch.zeros(V, dtype=torch.bool, device=device)
    is_trig[trigger_idx] = True
    tau_idx = all_idx[~is_trig]                                    # (N,)  N = V-K

    D_T = _trigger_indicator(V, trigger_idx, device, dtype)
    D_Tbar = 1.0 - D_T

    Q_rowsum = Q.sum(dim=1)                                        # (V,)
    Q_T_rowsum = (Q * D_T[None, :]).sum(dim=1)                     # (V,)
    Gamma_rowsum = Gamma.sum(dim=1)                                # (V,)
    Gamma_T_rowsum = (Gamma * D_T[None, :]).sum(dim=1)             # (V,)

    denom = V * (1 + rho)
    alpha0 = Q_rowsum[trigger_idx] / denom                         # (K,)
    alpha1 = 1.0 / denom
    beta0 = Q_T_rowsum[trigger_idx] / V - rho * alpha0             # (K,)
    beta1 = -rho * alpha1

    gamma0 = Gamma_rowsum[tau_idx] / denom                         # (N,)
    gamma1 = 1.0 / denom
    delta0 = Gamma_T_rowsum[tau_idx] / V - rho * gamma0            # (N,)
    delta1 = -rho * gamma1
    eps0 = -rho * Gamma_rowsum[tau_idx]                            # (N,)
    eps1 = 1.0
    phi0 = -rho * (1 + rho) * Gamma_T_rowsum[tau_idx] + rho ** 2 * Gamma_rowsum[tau_idx]
    phi1 = -rho

    _, _, EX, EY, EXY = _xy_moments(Q, Gamma, D_Tbar, trigger_idx, tau_idx, K, V)

    a0, a1 = alpha0[:, None], alpha1
    b0, b1 = beta0[:, None], beta1
    term_AC = _bilinear_expectation(a0, a1, gamma0[None, :], gamma1, EX, EY, EXY)
    term_AD = _bilinear_expectation(a0, a1, delta0[None, :], delta1, EX, EY, EXY)
    term_BE = _bilinear_expectation(b0, b1, eps0[None, :], eps1, EX, EY, EXY)
    term_BF = _bilinear_expectation(b0, b1, phi0[None, :], phi1, EX, EY, EXY)

    bar_AC = term_AC.mean(dim=1)   # (K,)
    bar_AD = term_AD.mean(dim=1)
    bar_BE = term_BE.mean(dim=1)
    bar_BF = term_BF.mean(dim=1)
    return bar_AC, bar_AD, bar_BE, bar_BF


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def expected_ontarget_logit(
    E: Tensor, P: Tensor, U: Tensor,
    W_QK1: Tensor, W_OV1: Tensor, W_QK2: Tensor, W_OV2: Tensor,
    trigger_idx: Tensor,
    beta: float,
    attn_norm_c: float,
    mu: Optional[Union[int, Tensor]] = None,
    reduce_triggers: bool = True,
) -> Tensor:
    """
    E_phi[h_{mu,tau} | tau_mu = a], averaged over the random output tau=phi(a)
    and (optionally) over triggers a in trigger_idx.

    Args:
        E: (d, V) embedding matrix.
        P: (d, L) positional-encoding matrix.
        U: (V, d) unembedding matrix.
        W_QK1, W_OV1, W_QK2, W_OV2: (d, d) attention/value weight matrices.
        trigger_idx: (K,) long tensor of trigger token indices into [0, V).
        beta: inverse temperature.
        attn_norm_c: attention normalization constant `c` (sec:transformer-definition).
        mu: query position(s), 0-indexed. int, 1-D LongTensor, or None (=> all
            valid positions 0..L-1, returned as the last tensor dimension).
        reduce_triggers: if True, average over a in trigger_idx (shape (...,));
            if False, keep the per-trigger axis first (shape (K, ...)).

    Returns:
        Tensor of shape () / (K,) if mu is a single int, or (L,) / (K,L) if
        mu is None, or (len(mu),) / (K,len(mu)) if mu is a 1-D tensor.
    """
    V = E.shape[1]
    L = P.shape[1]
    K = trigger_idx.numel()
    rho = K / V
    trigger_idx = trigger_idx.to(device=E.device, dtype=torch.long)

    dm = build_derived_matrices(E, P, U, W_QK1, W_OV1, W_QK2, W_OV2)
    S0, S1, S2, S3 = positional_sums(dm.M, V, rho)                 # each (L,)
    K1 = adjacent_position_correction_weight(dm.M, V, rho)          # (L,)
    bar_AC, bar_AD, bar_BE, bar_BF = _trigger_bar_terms(dm.Q, dm.Gamma, trigger_idx, V, rho)  # each (K,)
    corr_term = adjacent_position_correction_term(dm.Q, dm.Gamma, trigger_idx, V, rho)  # (K,)

    if mu is None:
        mu_S0, mu_S1, mu_S2, mu_S3, mu_K1 = S0, S1, S2, S3, K1      # (L,)
    else:
        mu_t = torch.as_tensor(mu, device=E.device, dtype=torch.long).reshape(-1)
        mu_S0, mu_S1, mu_S2, mu_S3 = S0[mu_t], S1[mu_t], S2[mu_t], S3[mu_t]
        mu_K1 = K1[mu_t]

    coef = rho * V * (1 + rho)
    # outer-product over (trigger, position)
    h = (
        bar_AC[:, None] * mu_S0[None, :]
        + coef * bar_AD[:, None] * mu_S1[None, :]
        - bar_BE[:, None] * mu_S2[None, :]
        - coef * bar_BF[:, None] * mu_S3[None, :]
        + corr_term[:, None] * mu_K1[None, :]                       # adjacent-position (n=1) correction
    ) * (beta / attn_norm_c)                                        # (K, len(mu) or L)

    if reduce_triggers:
        h = h.mean(dim=0)                                           # (len(mu) or L,)

    if isinstance(mu, int):
        h = h[..., 0]
    return h


# __all__ = [
#     "DerivedMatrices",
#     "build_derived_matrices",
#     "positional_sums",
#     "expected_ontarget_logit",
# ]