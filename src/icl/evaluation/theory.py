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
    snr = q * (d / eta.clamp(min=1e-8)) ** 0.5
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


# ══════════════════════════════════════════════════════════════════════════════
#  Order parameter extraction from weight matrices
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_order_parameters(
    A1:  Tensor,
    WV1: Tensor,
    A2:  Tensor,
    WV2: Tensor,
    U:   Tensor,
    pos_embeddings: Tensor,
    n_pairs: int = 200,
) -> dict[str, Tensor]:
    """
    Compute all five order parameters from the model weight matrices.

    Args:
        A1  : W_K1^T @ W_Q1,  shape (d, d)
        WV1 : W_V^(1),         shape (r, d)
        A2  : W_K2^T @ W_Q2,  shape (d, d)
        WV2 : W_V^(2),         shape (r, d)
        U   : unembedding,     shape (V, d)
        pos_embeddings : all positional encodings, shape (L, d)
        n_pairs : number of random adjacent pairs to average m over

    Returns:
        dict with keys: m, sigma1, q, eta, gamma
    """
    d = A1.shape[0]
    L = pos_embeddings.shape[0]

    # ── m : average p_s^T A1 p_{s-1} over adjacent pairs ─────────────────
    # sample n_pairs random adjacent positions
    idx = torch.randint(1, L, (n_pairs,))           # s in {1,...,L-1}
    p_s   = pos_embeddings[idx]                     # (n_pairs, d)
    p_sm1 = pos_embeddings[idx - 1]                 # (n_pairs, d)
    # p_s^T A1 p_{s-1} for each pair, then average
    m_vals = (p_s @ A1 * p_sm1).sum(dim=-1)         # (n_pairs,)
    m = m_vals.mean()

    # ── sigma1 : ||A1||_F / sqrt(d) ───────────────────────────────────────
    sigma1 = A1.norm(p="fro") / (d ** 0.5)

    # ── M = A2 @ WV1^T  (shape d x d, but computed efficiently) ──────────
    # We never form the full d x d matrix if r << d.
    # tr(A2 @ WV1) = tr(WV1 @ A2)  [cyclic] and both are tractable.
    # q = tr(A2 @ WV1) / d
    # For the trace: tr(A2 @ WV1) where A2: (d,d), WV1: (r,d)
    # = sum_i (A2 @ WV1^T)_{ii}  -- but A2 @ WV1^T is (d, r), not square.
    # Correct: M = A2 @ WV1^T is (d,r); we need tr of the (d,d) matrix
    # M_full = WV1^T @ A2 @ WV1 is (d,d) -- but that's not what we want.
    # We want M = A2 WV1 where WV1: (r,d), so A2: (d,d), M: (d,r) -- not square.
    # The scalar q = tr(M_square) where M_square = WV1^T A2^T... let's be careful:
    #
    # From the paper: M = A2 WV1 in R^{d x d}.
    # WV1 in R^{r x d}, so to get M in R^{d x d} we need WV1^T: (d, r) then
    # A2 (d,d) @ WV1^T (d,r) gives (d,r) -- still not square.
    #
    # The paper defines M = A2 W_V^(1) where W_V^(1) in R^{r x d}.
    # The composite acts as: e_v^T A2 W_V^(1) e_{v'} -- here W_V^(1) maps
    # R^d -> R^r and A2 maps R^d -> R^d, so the product A2: R^d->R^d composed
    # with W_V^(1)^T: R^r->R^d gives W_V^(1)^T A2^T or A2 W_V^(1)^T.
    #
    # Let's re-read: c_j = WV1 e_{v_{j-1}} in R^r, and the score is
    # e_{v_s}^T A2 c_j = e_{v_s}^T A2 WV1^T e_{v_{j-1}}  [WV1: r x d]
    # so M = A2 WV1^T in R^{d x d}.  q = tr(M)/d = tr(A2 WV1^T)/d.

    M = A2 @ WV1.T                                  # (d, d)

    # ── q = tr(M) / d ─────────────────────────────────────────────────────
    q = M.diagonal().sum() / d                      # = tr(M)/d

    # ── eta = ||M||_F^2 / d ───────────────────────────────────────────────
    eta = M.pow(2).sum() / d                        # = ||M||_F^2 / d

    # ── gamma = tr(U @ WV2^T) / d ─────────────────────────────────────────
    # U: (V, d), WV2: (r, d), WV2^T: (d, r)
    # U @ WV2^T: (V, r) -- not square, trace not directly defined.
    # From the paper: gamma = tr(U W_V^(2)^T)/d where U W_V^(2)^T in R^{V x r}.
    # The trace of a non-square matrix A is defined as sum of diagonal entries
    # of A, i.e. sum_i A_{ii} for i = 1..min(V,r).
    # More carefully from the derivation: gamma averages u_v^T WV2 e_v over v,
    # which after concentration gives sum_v (WV2)_{ii} / d -- the scalar is:
    # gamma = (1/d) * sum_{i=1}^{min(V,r)} (U WV2^T)_{ii}
    # = (1/d) * tr_{rect}(U WV2^T)
    # In practice we compute the full (V,r) product and take the partial trace.
    UWV2T = U @ WV2.T                               # (V, r)
    min_dim = min(U.shape[0], WV2.shape[0])
    gamma = UWV2T.diagonal()[:min_dim].sum() / d

    return {
        "m":      m,
        "sigma1": sigma1,
        "q":      q,
        "eta":    eta,
        "gamma":  gamma,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Training loop integration helper
# ══════════════════════════════════════════════════════════════════════════════

class LossEffTracker:
    """
    Convenience wrapper to call at each logging step during training.

    Usage:
        tracker = LossEffTracker(d=128, L=32, V=64, K=4)

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                ops = extract_order_parameters(A1, WV1, A2, WV2, U, pos_emb)
                record = tracker.step(ops, actual_loss=loss.item())
                print(record)

        df = tracker.to_dataframe()
    """

    def __init__(
        self,
        d: int,
        L: int,
        V: int,
        K: int,
        pi: float | None = None,
    ):
        self.d  = d
        self.L  = L
        self.V  = V
        self.K  = K
        self.pi = pi
        self.history: list[dict] = []

    def step(
        self,
        order_params: dict[str, Tensor],
        actual_loss: float | None = None,
        step: int | None = None,
    ) -> dict:
        """
        Evaluate L_eff and log everything.

        Args:
            order_params : dict with keys m, sigma1, q, eta, gamma
                           (output of extract_order_parameters)
            actual_loss  : the actual training/eval loss at this step
            step         : global step index

        Returns:
            dict with all quantities for immediate inspection
        """
        components = loss_eff(
            **order_params,
            d=self.d,
            L=self.L,
            V=self.V,
            K=self.K,
            pi=self.pi,
            return_components=True,
        )

        record = {
            "step":        step,
            "L_actual":    actual_loss,
            "L_eff":       components["L_eff"].item(),
            "P_prev":      components["P_prev"].item(),
            "P_ind":       components["P_ind"].item(),
            "delta_L":     components["delta_L"].item(),
            "SNR_layer1":  components["SNR_layer1"].item(),
            "SNR_layer2":  components["SNR_layer2"].item(),
            "m":           order_params["m"].item(),
            "sigma1":      order_params["sigma1"].item(),
            "q":           order_params["q"].item(),
            "eta":         order_params["eta"].item(),
            "gamma":       order_params["gamma"].item(),
        }

        if actual_loss is not None:
            record["residual"] = actual_loss - record["L_eff"]

        self.history.append(record)
        return record

    def to_dataframe(self):
        """Convert history to a pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame(self.history)
        except ImportError:
            raise ImportError("pandas required for to_dataframe(). "
                              "pip install pandas")

    def correlation(self) -> float:
        """
        Pearson correlation between L_eff and L_actual across all logged steps.
        Returns nan if actual_loss was never provided.
        """
        import math
        actual = [r["L_actual"] for r in self.history if r["L_actual"] is not None]
        pred   = [r["L_eff"]    for r in self.history if r["L_actual"] is not None]
        if len(actual) < 2:
            return float("nan")
        n  = len(actual)
        ma = sum(actual) / n
        mp = sum(pred) / n
        num = sum((a - ma) * (p - mp) for a, p in zip(actual, pred))
        da  = sum((a - ma) ** 2 for a in actual) ** 0.5
        dp  = sum((p - mp) ** 2 for p in pred) ** 0.5
        if da < 1e-12 or dp < 1e-12:
            return float("nan")
        return num / (da * dp)
