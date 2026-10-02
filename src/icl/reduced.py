"""The reduced backend: `MinimalTransformer` trained by SGD, rewritten in the
coordinates the loss actually depends on.

Only `WQK1`, `WQK2`, `WOV2` are trained; E, P, U, `WOV1` stay at their random
init. The logits depend on the weights only through the composed matrices

    M = P WQK1ᵀ Pᵀ         (L, L)
    Q = E WQK2ᵀ WOV1 Eᵀ    (V, V)
    G = U WOV2 Eᵀ          (V, V)

and one SGD step on the weights moves them *exactly* by

    M <- M - eta K_P  dM K_P,     K_P = P Pᵀ
    Q <- Q - eta K_E  dQ K_R,     K_E = E Eᵀ,  K_R = E WOV1ᵀ WOV1 Eᵀ
    G <- G - eta K_U  dG K_E,     K_U = U Uᵀ

with dM = dLoss/dM etc. Momentum and weight decay are linear in the weights, so
they map across too; Adam is not, and needs the full model.

Everything is stored normalised by sqrt(d): m = M/√d, q = Q/√d, g = G/√d and
k = K/d. (m, q, g) are the paper's M, Q, Gamma, and what
`matrices()` returns for both backends. In those variables the
forward pass has no d in it at all,

    A1     = mask1(m) / √L                  (L, L)
    H      = A1 @ onehot(x)                 (B, L, V)
    A2     = mask2(q[x] @ Hᵀ) / √L          (B, L, L)
    logits = beta * A2 @ gᵀ[x]              (B, L, V)

and the update is `m <- m - eta_0 k_P dm k_P` with `eta_0 = lr * d = alpha_lr`.
A finite d only enters through the Gram matrices k and the initial m, q, g, so
the cost of a step is O(B L² V), independent of d, and d = inf is just k = I.

Two initialisations:
- `from_full(model)`: compose an initialised `MinimalTransformer`. Same random
  draw as a full-model run with the same seed; needs the d x d weights once.
- `sample(args, d)`: draw the Gram matrices (Wishart, Bartlett decomposition)
  and the initial m, q, g directly from their joint law. Any d >= max(L, V),
  including `d = math.inf`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import make_mask

KERNELS = ("kP", "kE", "kR", "kU")


class ReducedTransformer(nn.Module):
    """Linear-attention `MinimalTransformer` in the (m, q, g) coordinates.

    Same `forward(x)` / `full_output(x)` / `matrices()` interface as
    the full model, so `compute_loss` and every probe work unchanged. Build it
    with `from_full` or `sample`; the constructor only allocates.
    """

    backend = "reduced"

    def __init__(self, args, d_model: float):
        super().__init__()
        if not args.lin_attn:
            raise ValueError("the reduced backend supports lin_attn=True only; use backend='full'")
        if args.dropout != 0.0:
            raise ValueError("the reduced backend supports dropout=0 only; use backend='full'")
        self.seq_len, self.vocab_size, self.d_model = args.seq_len, args.vocab_size, d_model
        self.beta, self.pred_mode = args.beta, args.pred_mode
        self.mask1_kind = getattr(args, "mask1", "causal")
        self.mask2_kind = getattr(args, "mask2", "causal")

        L, V = self.seq_len, self.vocab_size
        self.m = nn.Parameter(torch.zeros(L, L))
        self.q = nn.Parameter(torch.zeros(V, V))
        self.g = nn.Parameter(torch.zeros(V, V))
        for name, n in zip(KERNELS, (L, V, V, V)):
            self.register_buffer(name, torch.eye(n))
        self.register_buffer("mask1", make_mask(self.mask1_kind, L), persistent=False)
        self.register_buffer("mask2", make_mask(self.mask2_kind, L), persistent=False)

    # ---- construction --------------------------------------------------------

    @classmethod
    def from_full(cls, model) -> "ReducedTransformer":
        """The reduced model with exactly the state of an (initialised) `MinimalTransformer`.

        Composed in float64 so the Gram matrices are not rounded by TF32; stored in
        the full model's dtype and on its device.
        """
        if not model.lin_attn:
            raise ValueError("the reduced backend supports lin_attn=True only")
        d = model.d_model
        with torch.no_grad():
            f64 = lambda t: t.detach().double()
            E, P, U = f64(model.embed.E.weight), f64(model.embed.P.weight), f64(model.unembed.U.weight)
            W1, W2 = f64(model.attn1.WQK.weight), f64(model.attn2.WQK.weight)
            Wo1, Wo2 = f64(model.attn1.WOV.weight), f64(model.attn2.WOV.weight)
            R = Wo1 @ E.T                                          # (d, V)
            state = {
                "m": P @ W1.T @ P.T, "q": E @ W2.T @ R, "g": U @ Wo2 @ E.T,
                "kP": P @ P.T, "kE": E @ E.T, "kR": R.T @ R, "kU": U @ U.T,
            }
        reduced = cls(_Args.of(model), d).to(device=E.device, dtype=model.embed.E.weight.dtype)
        reduced._load({k: v / (math.sqrt(d) if k in ("m", "q", "g") else d) for k, v in state.items()})
        return reduced

    @classmethod
    def from_matrices(cls, mats: dict, args, device=None, dtype=None) -> "ReducedTransformer":
        """The model whose forward pass is exactly the one defined by the composed
        matrices `mats = {"M", "Q", "G"}` (the `matrices()` convention; torch
        tensors or numpy arrays), for any run trained with lin_attn, whatever its
        backend or d. `args` is the run's ModelArgs.

        Meant for evaluation: the Gram matrices are set to the identity (d = inf),
        which does not enter the forward pass but does enter `ReducedSGD`, so
        training it further is the d = inf dynamics. dtype defaults to that of M.
        """
        M, Q, G = (torch.as_tensor(mats[k]) for k in ("M", "Q", "G"))
        L, V = args.seq_len, args.vocab_size
        if M.shape != (L, L) or Q.shape != (V, V) or G.shape != (V, V):
            raise ValueError(f"matrices of shape {tuple(M.shape)}, {tuple(Q.shape)}, {tuple(G.shape)} "
                             f"do not match seq_len={L}, vocab_size={V}")
        reduced = cls(args, math.inf).to(device=device, dtype=dtype or M.dtype)
        reduced._load({"m": M, "q": Q, "g": G})
        return reduced

    @classmethod
    def sample(cls, args, d_model: float, sigma_0: float = 1.0, device=None,
               dtype=torch.float32) -> "ReducedTransformer":
        """Draw the Gram matrices and the initial (m, q, g) from the law that
        `MinimalTransformer.initialize_model` induces, without the d x d weights.

        With E, P, U ~ N(0, 1) and every W ~ N(0, sigma_0²/d): write each Gram
        matrix through its Bartlett factor a (K/d = a aᵀ, a lower triangular);
        the Gaussian weights then only appear as i.i.d. N(0, 1) sandwiches:

            m = s a_P Z_M a_Pᵀ,   q = s² a_E Z_Q (a_E a_R)ᵀ,   g = s a_U Z_G a_Eᵀ
            k_P = a_P a_Pᵀ,  k_E = a_E a_Eᵀ,  k_U = a_U a_Uᵀ,  k_R = s² (a_E a_R)(a_E a_R)ᵀ

        where s = sigma_0 and a_R is the Bartlett factor of N Nᵀ/d for the
        (d, V) i.i.d. matrix N with WOV1 Eᵀ = (s/√d) N (a_E √d)ᵀ in law. At
        d = inf every a is the identity. Uses the global torch RNG (seed with
        `set_seed`); not the same draw as `from_full`, the same distribution.
        """
        L, V = args.seq_len, args.vocab_size
        if not math.isinf(d_model) and d_model < max(L, V):
            raise ValueError(f"init='sample' needs d_model >= max(L, V) = {max(L, V)}; use init='full'")
        reduced = cls(args, d_model).to(device=device, dtype=dtype)
        kw = {"dtype": torch.float64, "device": device}
        a_P, a_E, a_U, a_R = (bartlett_factor(n, d_model, **kw) for n in (L, V, V, V))
        Z_M, Z_Q, Z_G = torch.randn(L, L, **kw), torch.randn(V, V, **kw), torch.randn(V, V, **kw)
        s = sigma_0
        a_ER = a_E @ a_R
        reduced._load({
            "m": s * a_P @ Z_M @ a_P.T, "q": s**2 * a_E @ Z_Q @ a_ER.T, "g": s * a_U @ Z_G @ a_E.T,
            "kP": a_P @ a_P.T, "kE": a_E @ a_E.T, "kR": s**2 * a_ER @ a_ER.T, "kU": a_U @ a_U.T,
        })
        return reduced

    def _load(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, value in state.items():
                getattr(self, name).copy_(value.to(getattr(self, name).dtype))

    # ---- forward -------------------------------------------------------------

    def _attn1(self) -> torch.Tensor:           # (L, L), == the full model's A1
        return self.m.masked_fill(~self.mask1, 0.0) / math.sqrt(self.seq_len)

    def full_output(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """{"A1", "A2", "logits"}; the attention maps equal the full model's.
        (The full model's X1, X2, S1, S2 live in d dimensions and have no
        counterpart here.)"""
        B, L = x.shape
        A1 = self._attn1()[:L, :L]
        H = A1 @ F.one_hot(x, self.vocab_size).to(A1.dtype)            # (B, L, V)
        if self.pred_mode == "last":
            # only the final query, attending to 0..L-1 through that row of the mask
            q_rows, m2 = self.q[x[:, -1:]], self.mask2[L - 1:L, :L]     # (B, 1, V), (1, L)
        else:
            q_rows, m2 = self.q[x], self.mask2[:L, :L]                 # (B, L, V), (L, L)
        A2 = (q_rows @ H.transpose(1, 2)).masked_fill(~m2, 0.0) / math.sqrt(L)
        logits = self.beta * (A2 @ self.g.T[x])                        # (B, Lq, V)
        return {"A1": A1.expand(B, -1, -1), "A2": A2, "logits": logits}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.full_output(x)["logits"]

    # ---- analysis objects ----------------------------------------------------

    def matrices(self) -> dict[str, torch.Tensor]:
        """Same keys and convention as `MinimalTransformer.matrices`: m, q, g on
        the CPU (m masked like the full model's M). Finite at d = inf."""
        with torch.no_grad():
            mats = {"M": self.m.masked_fill(~self.mask1, 0.0), "Q": self.q, "G": self.g}
            return {name: mat.detach().cpu() for name, mat in mats.items()}


class ReducedSGD(torch.optim.Optimizer):
    """SGD on (WQK1, WQK2, WOV2) as it acts on the reduced model's (m, q, g).

    `lr` is eta_0 = lr_full * d (i.e. `optim_args.alpha_lr`). Weight decay
    `weight_decay` is the full model's lambda: it enters as (lambda / d) * m, so it
    vanishes at d = inf. Momentum is torch.optim.SGD's (no dampening, no Nesterov).
    """

    def __init__(self, model: ReducedTransformer, lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
        wd = 0.0 if math.isinf(model.d_model) else weight_decay / model.d_model
        groups = [
            {"params": [model.m], "left": model.kP, "right": model.kP},
            {"params": [model.q], "left": model.kE, "right": model.kR},
            {"params": [model.g], "left": model.kU, "right": model.kE},
        ]
        super().__init__(groups, {"lr": lr, "momentum": momentum, "weight_decay": wd})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                direction = group["left"] @ p.grad @ group["right"]
                if group["weight_decay"]:
                    direction.add_(p, alpha=group["weight_decay"])
                if group["momentum"]:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = direction.clone()
                    else:
                        state["momentum_buffer"].mul_(group["momentum"]).add_(direction)
                    direction = state["momentum_buffer"]
                p.add_(direction, alpha=-group["lr"])
        return loss


def bartlett_factor(n: int, d: float, dtype=torch.float64, device=None) -> torch.Tensor:
    """Lower-triangular a with a aᵀ ~ Wishart(d, I_n) / d (Bartlett decomposition).

    It is the Cholesky factor of X Xᵀ / d for X an (n, d) N(0, 1) matrix; the
    identity at d = inf.
    """
    if math.isinf(d):
        return torch.eye(n, dtype=dtype, device=device)
    df = d - torch.arange(n, dtype=dtype)                            # d, d-1, ..., d-n+1
    diag = torch.distributions.Chi2(df).sample().sqrt()
    a = torch.randn(n, n, dtype=dtype).tril(-1) + torch.diag(diag)
    return (a / math.sqrt(d)).to(device)


class _Args:
    """The ModelArgs fields ReducedTransformer reads, taken from a full model."""

    @staticmethod
    def of(model) -> "_Args":
        a = _Args()
        a.seq_len, a.vocab_size, a.lin_attn = model.seq_len, model.vocab_size, model.lin_attn
        a.dropout, a.beta, a.pred_mode = model.drop, model.beta, model.pred_mode
        a.mask1, a.mask2 = model.mask1, model.mask2
        return a
