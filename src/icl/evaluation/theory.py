"""
Effective loss L_eff for the induction head emergence theory.

    L_eff = log(V) - (K/V) * pi * Phi(m * sqrt(d) / sigma1)^(L-1)
                                 * Phi(q * sqrt(d / eta))^(L-1)
                                 * DeltaL(gamma)

where:
    DeltaL(gamma) = log(V) - log(1 + (V-1) * exp(-gamma))

Order parameters:
    m      : layer-1 positional attention signal   = p_s^T A1 p_{s-1}          (scalar)
    sigma1 : layer-1 noise level                   = ||A1||_F / sqrt(d)        (scalar)
    q      : layer-2 induction signal              = tr(A2 @ WV1) / d          (scalar)
    eta    : layer-2 induction noise               = ||A2 @ WV1||_F^2 / d      (scalar)
    gamma  : readout alignment                     = tr(U @ WV2^T) / d         (scalar)

All order parameters can be passed as scalars or as batched tensors of shape (T,)
for vectorised evaluation over a trajectory of T checkpoints.
"""

import torch
from torch import Tensor
from typing import Union

# ── type alias ────────────────────────────────────────────────────────────────
FloatLike = Union[float, Tensor]


# ══════════════════════════════════════════════════════════════════════════════
#  Core building blocks
# ══════════════════════════════════════════════════════════════════════════════

def _phi(x: Tensor) -> Tensor:
    """Standard normal CDF Phi(x), numerically stable for large |x|."""
    return 0.5 * (1.0 + torch.erf(x / (2.0 ** 0.5)))


def P_prev(m: Tensor, sigma1: Tensor, d: int, L: int) -> Tensor:
    """
    Probability that layer-1 (previous-token head) attends correctly at all
    positions.

        P_prev = Phi(m * sqrt(d) / sigma1)^(L-1)

    Computed in log-space to avoid underflow for large L.

    Args:
        m      : layer-1 signal order parameter
        sigma1 : layer-1 noise order parameter  (must be > 0)
        d      : embedding dimension
        L      : sequence length

    Returns:
        Tensor of same shape as m / sigma1.
    """
    snr = m * (d ** 0.5) / sigma1.clamp(min=1e-8)
    log_p = (L - 1) * torch.log(_phi(snr).clamp(min=1e-40))
    return torch.exp(log_p)


def P_ind(q: Tensor, eta: Tensor, d: int, L: int) -> Tensor:
    """
    Probability that layer-2 (induction head) attends to the correct key.

        P_ind = Phi(q * sqrt(d / eta))^(L-1)

    Args:
        q   : layer-2 signal order parameter
        eta : layer-2 noise order parameter  (must be > 0)
        d   : embedding dimension
        L   : sequence length

    Returns:
        Tensor of same shape as q / eta.
    """
    snr = q * (d ** 0.5) / eta.clamp(min=1e-8)
    log_p = (L - 1) * torch.log(_phi(snr).clamp(min=1e-40))
    return torch.exp(log_p)


def delta_L(gamma: Tensor, V: int) -> Tensor:
    """
    Induction gain over random guessing:

        DeltaL(gamma) = log(V) - log(1 + (V-1) * exp(-gamma))
                      = log(V) - log(1 + (V-1)*exp(-gamma))

    Computed via log-sum-exp for numerical stability.

    When gamma >> log(V-1): DeltaL -> log(V)   (perfect prediction)
    When gamma  = 0       : DeltaL  = 0         (same as random)
    When gamma  < 0       : DeltaL  < 0         (worse than random)

    Args:
        gamma : readout order parameter
        V     : vocabulary size

    Returns:
        Tensor of same shape as gamma.
    """
    log_V = torch.log(torch.tensor(float(V)))
    # log(1 + (V-1)*exp(-gamma)) computed stably
    # = log(exp(0) + (V-1)*exp(-gamma))
    # use torch.logaddexp: log(exp(a) + exp(b))
    log_denom = torch.logaddexp(
        torch.zeros_like(gamma),                          # log(exp(0)) = 0
        torch.log(torch.tensor(float(V - 1))) - gamma,   # log((V-1)*exp(-gamma))
    )
    return log_V - log_denom


def pi_theory(L: int, V: int) -> float:
    """
    Fraction of trigger positions that have a prior occurrence in context.

        pi = 1 - V/(L-1) * [(1-1/V) - (1-1/V)^L]

    This is the position-averaged repeat probability (exact formula from
    Appendix B of the paper).

    Args:
        L : sequence length
        V : vocabulary size

    Returns:
        float in [0, 1]
    """
    p = 1.0 - 1.0 / V          # (V-1)/V
    # average of [1 - p^(s-1)] for s = 2,...,L
    # = 1 - (1/(L-1)) * sum_{s=2}^{L} p^{s-1}
    # = 1 - (1/(L-1)) * p * (1 - p^{L-1}) / (1 - p)
    # = 1 - V/(L-1) * (p - p^L)
    if L <= 1:
        return 0.0
    return 1.0 - (V / (L - 1)) * (p - p ** L)


# ══════════════════════════════════════════════════════════════════════════════
#  Main function
# ══════════════════════════════════════════════════════════════════════════════

def loss_eff(
    m:      FloatLike,
    sigma1: FloatLike,
    q:      FloatLike,
    eta:    FloatLike,
    gamma:  FloatLike,
    *,
    d: int,
    L: int,
    V: int,
    K: int,
    pi: float | None = None,
    return_components: bool = False,
) -> Tensor | dict[str, Tensor]:
    """
    Effective loss predicted by the statistical mechanics theory:

        L_eff = log(V)
              - (K/V) * pi * P_prev(m, sigma1) * P_ind(q, eta) * DeltaL(gamma)

    Args:
        m      : layer-1 positional attention signal  [scalar or (T,)]
        sigma1 : layer-1 noise level                  [scalar or (T,)]
        q      : layer-2 induction signal             [scalar or (T,)]
        eta    : layer-2 noise variance               [scalar or (T,)]
        gamma  : readout alignment                    [scalar or (T,)]

        d      : embedding dimension                  (int, keyword-only)
        L      : sequence length                      (int, keyword-only)
        V      : vocabulary size                      (int, keyword-only)
        K      : number of trigger tokens             (int, keyword-only)
        pi     : repeat-occurrence probability; if None, computed from L and V
                 via pi_theory(L, V)                  (float, keyword-only)
        return_components : if True, return a dict with all intermediate
                            quantities for diagnosis  (bool, keyword-only)

    Returns:
        Tensor of same shape as the input order parameters, or dict if
        return_components=True.
    """
    # ── cast everything to tensors on the same device ─────────────────────
    def _t(x: FloatLike) -> Tensor:
        if isinstance(x, Tensor):
            return x.float()
        return torch.tensor(float(x))

    m      = _t(m)
    sigma1 = _t(sigma1)
    q      = _t(q)
    eta    = _t(eta)
    gamma  = _t(gamma)

    # broadcast to common shape
    m, sigma1, q, eta, gamma = torch.broadcast_tensors(m, sigma1, q, eta, gamma)

    # ── task constants ─────────────────────────────────────────────────────
    log_V  = torch.log(torch.tensor(float(V)))
    eps    = K / V                                  # trigger frequency
    pi_val = pi_theory(L, V) if pi is None else pi

    # ── circuit success probabilities ──────────────────────────────────────
    pp = P_prev(m, sigma1, d, L)                    # P_prev
    pi_ind = P_ind(q, eta, d, L)                    # P_ind

    # ── readout gain ───────────────────────────────────────────────────────
    dL = delta_L(gamma, V)                          # DeltaL(gamma)
    # print(f"Debug: P_prev={pp}, P_ind={pi_ind}, DeltaL={dL}, eps={eps}, pi={pi_val}")
    # ── effective loss ─────────────────────────────────────────────────────
    L_eff = log_V - eps * pi_val * pp * pi_ind * dL

    if not return_components:
        return L_eff

    # ── diagnostic breakdown ───────────────────────────────────────────────
    snr1 = m * (d ** 0.5) / sigma1.clamp(min=1e-8)
    snr2 = q * (d / eta.clamp(min=1e-8)) ** 0.5

    return {
        "L_eff":           L_eff,
        "log_V":           log_V.expand_as(L_eff),
        "P_prev":          pp,
        "P_ind":           pi_ind,
        "delta_L":         dL,
        "SNR_layer1":      snr1,
        "SNR_layer2":      snr2,
        "pi":              torch.tensor(pi_val).expand_as(L_eff),
        "gain_term":       eps * pi_val * pp * pi_ind * dL,
    }

