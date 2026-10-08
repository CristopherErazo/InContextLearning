"""Linear forms in the summary statistics, and their evaluation given S_mu or given ell.

Every piece of a key-set sum, of the column energy and of a bracket is `coef * X * Y` (or
`coef * X`) with X, Y linear forms in the statistics of the sequence (tables
summary_stats and noise_stats of the appendix). Given S_mu (`GivenS`) a form is evaluated
on the measured or sampled statistics. Given ell (`GivenEll`) the piece is replaced by its
conditional mean, E[X Y | ell] = E X E Y + Cov(X, Y), from the moments of tables moments,
cov and eq. noise_moments. One definition of the pieces therefore serves both levels.

The statistics (m_nu = the mean previous-token entry of key nu, M_on phi(nu/L)):

    target ("t", one per row)                        non-target ("b", one per row and b)
    ell     earlier a's                              c       occurrences c_b
    f       free targets                             Ub      sum of m over them   (M_on U^phi_b)
    Nm      sum of m over the witness keys (M_on N^phi)  Utr  ... over the triggered ones (M_on U^{phi,tr}_b)
    Fm      sum of m over the free target keys       Wb, Pb  W_bar_b, P_bar_b
    W, P, R                                          Yb      (M_on)^2 Y_b (profile-weighted pairs)
    Nm2     sum of m^2 over the witness keys         pairs_b c_b (c_b - 1)
    Rm      sum of m_w (mu - 1 - w) over witness keys (M_on R^phi)
    X1, X2  eq. X12                                  UbN2    sum of m^2 over the keys of b with a
    pairs_phi  (ell + f)(ell + f - 1)                        non-trigger predecessor
    Fm2     sum of m^2 over the free target keys

Fm2 and UbN2 are the same-source pairs nu = nu' of kappa_NN g^N_I g^N_I (item b1 of the
audit), removed from the token-coherent noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .table import Bilinear

T_STATS = ("ell", "f", "Nm", "Fm", "W", "P", "R", "Nm2", "Rm", "X1", "X2", "pairs_phi", "Fm2")
B_STATS = ("c", "Ub", "Utr", "Wb", "Pb", "Yb", "pairs_b", "UbN2")


class Form:
    """sum_i coefs[i] * stat_i + const; coefficients are numbers or (rows,) tensors."""
    __slots__ = ("coefs", "const")

    def __init__(self, coefs: dict | None = None, const=0.0):
        self.coefs, self.const = dict(coefs or {}), const

    @classmethod
    def of(cls, name: str) -> "Form":
        if name not in T_STATS and name not in B_STATS:
            raise ValueError(f"unknown statistic {name!r}")
        return cls({name: 1.0})

    @property
    def kind(self) -> str:
        kinds = {"t" if name in T_STATS else "b" for name in self.coefs}
        if len(kinds) > 1:
            raise ValueError(f"a form mixes target and non-target statistics: {list(self.coefs)}")
        return kinds.pop() if kinds else "c"

    def __add__(self, other):
        if not isinstance(other, Form):
            return Form(self.coefs, self.const + other)
        coefs = dict(self.coefs)
        for name, coef in other.coefs.items():
            coefs[name] = coefs[name] + coef if name in coefs else coef
        return Form(coefs, self.const + other.const)

    __radd__ = __add__

    def __neg__(self):
        return Form({name: -coef for name, coef in self.coefs.items()}, -self.const)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, factor):
        return Form({name: factor * coef for name, coef in self.coefs.items()}, factor * self.const)

    __rmul__ = __mul__


@dataclass
class Piece:
    """coef * (X Y + linear) (or coef * X): one summand of a key-set sum or of the energy.
    cross=True: X is read at b and Y at b' (a b-b' sum, evaluated as a Bilinear)."""
    mechanism: str
    detail: str
    params: tuple
    coef: object
    X: Form
    Y: Form | None = None
    cross: bool = False
    linear: Form | None = None


def _as_column(value, kind: str):
    """A (rows,) tensor as a column (rows, 1) when it multiplies non-target values."""
    if kind == "b" and torch.is_tensor(value) and value.dim() == 1:
        return value[:, None]
    return value


def _product_kind(*kinds) -> str:
    return "b" if "b" in kinds else "t"


class _Evaluator:
    rows: int
    nb: int
    dtype: torch.dtype
    device: torch.device

    def stat(self, name: str) -> torch.Tensor:
        raise NotImplementedError

    def form(self, form: Form, kind: str | None = None) -> torch.Tensor:
        """The value (rows,) or (rows, nb) of a form: given S its value, given ell its mean."""
        kind = kind or form.kind
        shape = (self.rows, self.nb) if kind == "b" else (self.rows,)
        value = torch.zeros(shape, dtype=self.dtype, device=self.device) + _as_column(
            torch.as_tensor(form.const, dtype=self.dtype, device=self.device), kind)
        for name, coef in form.coefs.items():
            value = value + _as_column(coef, kind) * self.stat(name)
        return value

    def cov(self, X: Form, Y: Form) -> torch.Tensor | float:
        return 0.0

    def piece(self, piece: Piece):
        if piece.Y is None:
            kind = _product_kind(piece.X.kind)
            return _as_column(piece.coef, kind) * self.form(piece.X, kind)
        if piece.cross:
            return self.cross(piece)
        kind = _product_kind(piece.X.kind, piece.Y.kind)
        X, Y = self.form(piece.X), self.form(piece.Y)
        if kind == "b":
            X, Y = _as_column(X, "b"), _as_column(Y, "b")
        value = X * Y + self.cov(piece.X, piece.Y)
        if piece.linear is not None:
            value = value + self.form(piece.linear, kind)
        return _as_column(piece.coef, kind) * value

    def cross(self, piece: Piece) -> Bilinear:
        return Bilinear(((piece.coef, self.form(piece.X, "b"), self.form(piece.Y, "b")),))


class GivenS(_Evaluator):
    """Evaluate the forms on the statistics of each row: {name: (rows,) or (rows, nb)}."""

    def __init__(self, stats: dict, nb: int):
        self.stats = stats
        first = stats["ell"]
        self.rows, self.nb, self.dtype, self.device = len(first), nb, first.dtype, first.device

    def stat(self, name):
        return self.stats[name]


def moment_tables(mu, ell, lam, r, s, window: dict, X1_mean, col) -> tuple[dict, dict]:
    """The means (table moments) and covariances (table cov, eq. noise_moments) of the
    statistics given ell. mu, ell, lam: per row; r: per row and b, the occurrence rates;
    s: per b, 1 for the outputs of other triggers; col lifts a per-row value to one per
    row and b. Plain arithmetic: tensors (`GivenEll`) or sympy (`symbolic`)."""
    P1, P2, Psi = window["Phi1"], window["Phi2"], window["Psi"]
    dPsi = Psi - P1 / 2
    means = {
        "ell": ell, "f": lam, "Nm": ell * P1, "Fm": lam * P1, "W": mu * (ell + lam) / 2,
        "P": ell * (ell - 1) / 2 + lam * ell / 2, "R": mu * ell / 2, "Nm2": ell * P2, "Rm": mu * ell * (P1 - Psi),
        "X1": X1_mean, "X2": mu * ell * (2 * ell + 1) / 6, "pairs_phi": ell ** 2 - ell + 2 * ell * lam + lam ** 2,
        "c": r, "Ub": r * col(P1), "Utr": s * col(lam * P1), "Wb": col(mu) * r / 2, "Pb": r * col(ell) / 2,
        "Yb": s * col(lam ** 2 * P1 ** 2), "pairs_b": r ** 2,
        # the keys with a non-trigger predecessor: rate Lambda for every token (an output's other half
        # follows its trigger), and the free targets
        "Fm2": lam * P2, "UbN2": 0 * r + col(lam * P2),
    }
    variance_P = lam * ell * (2 * ell + 1) / 6 + lam ** 2 * ell / 12
    c_mu, c_ell, c_lam = col(mu), col(ell), col(lam)
    covariances = {
        ("Nm", "Nm"): ell * (P2 - P1 ** 2), ("Fm", "Fm"): lam * P2, ("W", "W"): mu ** 2 * (ell / 12 + lam / 3),
        ("P", "P"): variance_P, ("R", "R"): mu ** 2 * ell / 12, ("f", "f"): lam,
        ("Nm", "W"): mu * ell * dPsi, ("Nm", "P"): -lam * ell * dPsi, ("Nm", "R"): -mu * ell * dPsi,
        ("Fm", "W"): mu * lam * Psi, ("Fm", "P"): lam * ell * Psi, ("W", "P"): mu * lam * ell / 4,
        ("W", "R"): -mu ** 2 * ell / 12, ("P", "R"): mu * lam * ell / 12,
        ("f", "Fm"): lam * P1, ("f", "W"): mu * lam / 2, ("f", "P"): lam * ell / 2,
        # target - non-target, through the positions of the earlier a's
        ("Nm", "Pb"): -r * col(ell * dPsi), ("W", "Pb"): -r * col(mu * ell) / 12,
        ("P", "Pb"): r * col(lam * ell) / 12, ("R", "Pb"): r * col(mu * ell) / 12,
        # within one non-target token
        ("Ub", "Ub"): r * col(P2), ("Utr", "Utr"): s * col(lam * P2), ("Wb", "Wb"): c_mu ** 2 * r / 3,
        ("Pb", "Pb"): r * c_ell * (2 * c_ell + 1) / 6 + r ** 2 * c_ell / 12, ("c", "c"): r,
        ("Ub", "Utr"): s * col(lam * P2), ("Ub", "Wb"): c_mu * r * col(Psi), ("Ub", "Pb"): r * col(ell * Psi),
        ("Ub", "c"): r * col(P1), ("Utr", "Wb"): s * col(mu * lam * Psi), ("Utr", "Pb"): s * col(lam * ell * Psi),
        ("Utr", "c"): s * col(lam * P1), ("Wb", "Pb"): c_mu * r * c_ell / 3, ("Wb", "c"): c_mu * r / 2,
        ("Pb", "c"): r * c_ell / 2,
    }
    return means, covariances


class GivenEll(_Evaluator):
    """Evaluate the forms by their means given ell, and products by E X E Y + Cov(X, Y),
    with the moments of tables moments and cov and eq. noise_moments of the appendix
    (A1-A3), the window moments read on the m-weighted profile:

        Phi1 = mean of m over the keys, Phi2 = mean of m^2, Psi = mean of m x
        (x the relative key position), dPsi = Psi - Phi1 / 2.

    r: (rows, nb) occurrence rates r_b, s: (nb,) 1 for the outputs of other triggers,
    lam: (rows,) Lambda, the rate of the free targets. P_bar is linked to the target
    statistics and to P_bar of another b' through the positions of the earlier a's."""

    def __init__(self, mu, ell, lam, r, s, window: dict, M_off, X1_mean):
        self.mu, self.ell, self.lam, self.r, self.s = mu, ell, lam, r, s
        self.rows, self.nb, self.dtype, self.device = len(mu), r.size(1), r.dtype, r.device
        self.means, self.covariances = moment_tables(mu, ell, lam, r, s.to(r.dtype), window, X1_mean,
                                                     col=lambda value: value[:, None])

    def stat(self, name):
        return self.means[name]

    def _cov(self, a: str, b: str):
        return self.covariances.get((a, b), self.covariances.get((b, a)))

    def cov(self, X: Form, Y: Form):
        kind = _product_kind(X.kind, Y.kind)
        total = 0.0
        for a, x in X.coefs.items():
            for b, y in Y.coefs.items():
                c = self._cov(a, b)
                if c is not None:
                    total = total + _as_column(x, kind) * _as_column(y, kind) * c
        return total

    def cross(self, piece: Piece) -> Bilinear:
        """E[X_b Y_b'] = E X_b E Y_b' + x_P y_P Cov(P_bar_b, P_bar_b'), the latter
        r_b r_b' ell / 12."""
        parts = [(piece.coef, self.form(piece.X, "b"), self.form(piece.Y, "b"))]
        x, y = piece.X.coefs.get("Pb"), piece.Y.coefs.get("Pb")
        if x is not None and y is not None:
            coef = torch.as_tensor(piece.coef, dtype=self.dtype, device=self.device) * self.ell / 12
            parts.append((coef, _as_column(x, "b") * self.r, _as_column(y, "b") * self.r))
        return Bilinear(tuple(parts))
