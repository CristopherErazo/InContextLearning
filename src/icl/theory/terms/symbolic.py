"""The terms at a trigger query given ell as sympy expressions: the same pieces as
`trigger_terms_ell` (`trigger._structure`, the moment tables of `forms`, the readout of
`trigger.assemble_trigger`), evaluated on symbols instead of tensors. For the power counting
and for writing the pruned formulas out.

    terms = symbolic_terms_ell()            # {name: (tags, expression)}
    terms["on,on/attention/C_phi.incoherent"][1]

The symbols (`SYMBOLS`): the order parameters (as in `ORDER_PARAMS` and `VARIANCES`; M_on
enters only through the window moments), the window moments Phi1, Phi2, Psi of
`window_moments`, the query position mu, its number of keys n = mu - 2, ell, the rate of the
free targets Lambda, beta, L, V, K, and for a non-target column its class s (1 for an output
of another trigger, 0 for a plain token; s2 for the second column of "b,b'"). With
`background="exact"` the sums D, G1, K1, K2, J2 of the mean M are the exact discrete sums
(`sums._key_sums`) written in n and the window moments, so that substituting the numbers of
`trigger_terms_ell(..., background="exact")` reproduces it; "leading" is eq. sums_bg.

`symbolic_terms_S(query)` is the level given S_mu: every term of `trigger_terms` or
`nontrigger_terms` as a formula in the statistics, the order parameters (with dQ, q2_T, dq2
as symbols), the rates and the deterministic sums (`S_SYMBOLS`), as the appendix writes
eqs. pairs_trigger and pairs_nontrigger; the b' column of a b-b' sum has the statistics
"<name>_2".
"""
from __future__ import annotations

import sympy as sp

from .forms import B_STATS, T_STATS, Form, Piece, _Evaluator, moment_tables
from .table import Bilinear
from .trigger import ELL_LABELS, KEYSETS_OF_PAIR, TRIGGER_PAIRS, _derived, _mean_terms, _readout_factors, \
    _spread_terms, _structure, assemble_trigger, coincidence_rates

NAMES = ("M_off", "Q_on", "Q_T", "G_on", "G_T", "var_M_on", "var_M_off", "var_Q_on", "var_Q_T", "var_G_on",
         "var_G_T", "var_G_N", "Phi1", "Phi2", "Psi", "mu", "n", "ell", "Lambda", "beta", "L", "V", "K", "s", "s2")
SYMBOLS = {name: sp.Symbol(name, real=True) for name in NAMES}


class SymbolicEll(_Evaluator):
    """`GivenEll` on symbols: one non-target column of class s (b' of class s2 in a
    b-b' sum)."""

    def __init__(self, symbols: dict, X1_mean):
        S = symbols
        self.ell, self.symbols = S["ell"], S
        window = {name: S[name] for name in ("Phi1", "Phi2", "Psi")}
        self.r = S["Lambda"] * (1 + S["s"])
        self.means, self.covariances = moment_tables(S["mu"], S["ell"], S["Lambda"], self.r, S["s"], window, X1_mean,
                                                     col=lambda value: value)

    def stat(self, name):
        return self.means[name]

    def form(self, form: Form, kind=None):
        return form.const + sum((coef * self.stat(name) for name, coef in form.coefs.items()), sp.Integer(0))

    def _cov(self, a, b):
        return self.covariances.get((a, b), self.covariances.get((b, a)))

    def cov(self, X: Form, Y: Form):
        total = 0.0
        for a, x in X.coefs.items():
            for b, y in Y.coefs.items():
                c = self._cov(a, b)
                if c is not None:
                    total = total + x * y * c
        return total

    def _second(self, value):
        S = self.symbols
        return sp.sympify(value).subs(S["s"], S["s2"])

    def cross(self, piece: Piece) -> Bilinear:
        """As `GivenEll.cross`: E X_b E Y_b' + x_P y_P r_b r_b' ell / 12, Y read at b'."""
        parts = [(piece.coef, self.form(piece.X), self._second(self.form(piece.Y)))]
        x, y = piece.X.coefs.get("Pb"), piece.Y.coefs.get("Pb")
        if x is not None and y is not None:
            parts.append((piece.coef * self.ell / 12, x * self.r, self._second(y * self.r)))
        return Bilinear(tuple(parts))


def background_sums(symbols: dict, kind: str = "exact") -> dict:
    """The deterministic sums D, G1, K1, K2, J2, Mall (sum of m over the keys), n and
    nn = n (n - 1) / 2 in the window moments: exact on the discrete positions (as
    `sums._key_sums`, with sum_j j m_j = n^2 Psi + n Phi1 / 2) or at leading order (eq.
    sums_bg, in mu, as `trigger._leading_background`)."""
    S = symbols
    n, mu, P1, P2, Psi, M_off = S["n"], S["mu"], S["Phi1"], S["Phi2"], S["Psi"], S["M_off"]
    vMon, vMoff = S["var_M_on"], S["var_M_off"]
    pairs = n * (n - 1) / 2
    exact = {"Mall": n * P1, "n": n, "nn": pairs}
    if kind == "leading":
        return exact | {"D": vMon * mu + vMoff * mu ** 2 / 2, "K1": mu * (P1 + M_off * mu / 2),
                        "J2": mu * (P2 + 2 * M_off * mu * (P1 - Psi) + M_off ** 2 * mu ** 2 / 3),
                        "G1": mu * (P2 + M_off ** 2 * mu / 2), "K2": mu * (P2 + 2 * M_off * mu * Psi + M_off ** 2 * mu ** 2 / 3)}
    if kind != "exact":
        raise ValueError(f"background must be 'exact' or 'leading', got {kind!r}")
    m1, m2, position_m = n * P1, n * P2, n ** 2 * Psi + n * P1 / 2
    squares = (n - 1) * n * (2 * n - 1) / 6
    return exact | {"D": n * vMon + pairs * vMoff, "G1": m2 + M_off ** 2 * pairs, "K1": m1 + M_off * pairs,
                    "K2": m2 + 2 * M_off * (position_m - m1) + M_off ** 2 * squares,
                    "J2": m2 + 2 * M_off * (n * m1 - position_m) + M_off ** 2 * squares}


def _expression(value) -> sp.Expr:
    """A term's value as one expression, its float constants (1.0, 2.0, -1.0 of the pieces)
    made rational."""
    if isinstance(value, Bilinear):
        value = sum((coef * u * v for coef, u, v in value.parts), sp.Integer(0))
    value = sp.sympify(value)
    return value.xreplace({number: sp.nsimplify(number, rational=True) for number in value.atoms(sp.Float)})


def symbolic_terms_ell(pairs=None, background: str = "exact", as_written: bool = False) -> dict:
    """{name: (tags, expression)} for every term of `trigger_terms_ell` (same names and tags),
    each a sympy expression in `SYMBOLS`. A non-target value is that of a column of class s;
    "b,b'" that of a pair of distinct columns of classes s, s2. pairs, as_written: as
    `trigger_terms_ell` (classes, pairs of `TRIGGER_PAIRS`)."""
    S = SYMBOLS
    p = _derived({name: S[name] for name in ("M_off", "Q_on", "Q_T", "G_on", "G_T", "var_M_on", "var_M_off",
                                              "var_Q_on", "var_Q_T", "var_G_on", "var_G_T", "var_G_N")})
    mu, ell, M_off = S["mu"], S["ell"], S["M_off"]
    evaluator = SymbolicEll(S, X1_mean=mu * ell * (S["Psi"] + M_off * mu / 3))
    structure = _structure(p, background_sums(S, background), coincidence_rates(S["V"], S["K"]), as_written)
    beta, L = S["beta"], S["L"]
    terms = _mean_terms(structure, evaluator, p, beta, L, ELL_LABELS["mean"], pairs)
    terms += _spread_terms(structure, evaluator, p, beta, L, pairs)
    needed = None if pairs is None else {keyset for pair in pairs for keyset in KEYSETS_OF_PAIR.get(pair, ())}
    evaluated = {"keysets": {name: [(piece, evaluator.piece(piece)) for piece in pieces]
                             for name, pieces in structure["keysets"].items() if needed is None or name in needed},
                 "energy": [(piece, evaluator.piece(piece)) for piece in structure["energy"]]
                 if needed is None or "E" in needed else [],
                 "own": {cls: (piece, evaluator.piece(piece)) for cls, piece in structure["own"].items()
                         if needed is None or "own" in needed}}
    terms += assemble_trigger(evaluated, _readout_factors(p, (beta / L) ** 2), ELL_LABELS, 0, pairs)
    tags = ("pair", "channel", "column", "mechanism", "keyset", "detail", "label", "params")
    return {term.name: ({tag: getattr(term, tag) for tag in tags}, _expression(term.value)) for term in terms}


# ---- given S --------------------------------------------------------------------------------

S_NAMES = ("M_off", "Q_on", "Q_T", "dQ", "q2_T", "dq2", "G_on", "G_T", "var_M_on", "var_M_off", "var_Q_on",
           "var_Q_T", "var_Q_N", "var_G_on", "var_G_T", "var_G_N", "kappa_2", "kappa_N", "kappa_NN", "p_T", "kappa_bi",
           "D", "G1", "K1", "K2", "J2", "Mall", "n", "nn", "beta", "L",
           *T_STATS, *B_STATS, *(f"{name}_2" for name in B_STATS))
S_SYMBOLS = {name: sp.Symbol(name, real=True) for name in S_NAMES}


class SymbolicS(_Evaluator):
    """`GivenS` on symbols: every statistic is its own symbol; a b-b' sum reads the second
    factor at b' (the statistics "<name>_2")."""

    def stat(self, name):
        return S_SYMBOLS[name]

    def form(self, form: Form, kind=None):
        return form.const + sum((coef * self.stat(name) for name, coef in form.coefs.items()), sp.Integer(0))

    def cov(self, X, Y):
        return 0.0

    def cross(self, piece: Piece) -> Bilinear:
        second = {S_SYMBOLS[name]: S_SYMBOLS[f"{name}_2"] for name in B_STATS}
        return Bilinear(((piece.coef, self.form(piece.X), sp.sympify(self.form(piece.Y)).xreplace(second)),))


def symbolic_terms_S(query: str = "trigger", pairs=None, as_written: bool = False) -> dict:
    """{name: (tags, expression)} for every term of `trigger_terms` ("trigger") or
    `nontrigger_terms` ("nontrigger") given S_mu, in `S_SYMBOLS` (same names and tags)."""
    from . import nontrigger, trigger
    S = S_SYMBOLS
    p = {name: S[name] for name in ("M_off", "Q_on", "Q_T", "dQ", "q2_T", "dq2", "G_on", "G_T", "var_M_on",
                                    "var_M_off", "var_Q_on", "var_Q_T", "var_Q_N", "var_G_on", "var_G_T", "var_G_N")}
    rates = {name: S[name] for name in ("kappa_2", "kappa_N", "kappa_NN", "p_T", "kappa_bi")}
    bg = {name: S[name] for name in ("D", "G1", "K1", "K2", "J2", "Mall", "n", "nn")}
    evaluator = SymbolicS()
    factors = _readout_factors(p, (S["beta"] / S["L"]) ** 2)
    if query == "trigger":
        structure = _structure(p, bg, rates, as_written)
        terms = _mean_terms(structure, evaluator, p, S["beta"], S["L"], trigger.S_LABELS["mean"], pairs)
        needed = trigger._needed_keysets(pairs)
        terms += assemble_trigger(trigger._evaluate(structure, evaluator, needed), factors, trigger.S_LABELS, 0, pairs)
    elif query == "nontrigger":
        structure = nontrigger._structure(p, bg, rates, as_written)
        terms = nontrigger.assemble_nontrigger(trigger._evaluate(structure, evaluator, nontrigger._needed(pairs)),
                                               factors, nontrigger.S_LABEL, 0, pairs)
    else:
        raise ValueError(f"query must be 'trigger' or 'nontrigger', got {query!r}")
    tags = ("pair", "channel", "column", "mechanism", "keyset", "detail", "label", "params")
    return {term.name: ({tag: getattr(term, tag) for tag in tags}, _expression(term.value)) for term in terms}


# ---- power counting --------------------------------------------------------------------------

# The dimensionless parameters of the power counting: every order parameter relative to the
# scale of its block in the signal (Phi1 for M, Q_on for Q, G_on for Gamma), with the window
# length mu absorbed where a sum over the keys carries it (r_M, s_Moff), and p_T = Lambda / mu.
RATIOS = {
    "r_M": "M_off mu / Phi1", "s_Moff": "var_M_off mu / Phi1^2", "s_Mon": "var_M_on / Phi1^2",
    "r_Q": "Q_T / Q_on", "s_Qon": "var_Q_on / Q_on^2", "s_QT": "var_Q_T / Q_on^2",
    "r_G": "G_T / G_on", "s_Gon": "var_G_on / G_on^2", "s_GT": "var_G_T / G_on^2", "s_GN": "var_G_N / G_on^2",
    "d2": "Phi2 / Phi1^2 - 1", "dPsi": "Psi / Phi1 - 1/2",
}
ORDER_SYMBOLS = {name: sp.Symbol(name, real=True) for name in RATIOS}
COUNT_SYMBOLS = {name: SYMBOLS[name] for name in ("ell", "Lambda")} | {"rho": sp.Symbol("rho", positive=True)}


def ratio_values(order_params: dict, window: dict, mu, V: int, K: int) -> dict:
    """The numbers of `RATIOS` (and rho = K / V, Lambda = p_T mu) at one query position:
    order_params as `trigger_terms_ell`, window = the window moments at mu (floats)."""
    get = lambda name: float(order_params.get(name, 0.0))
    P1 = window["Phi1"]
    return {"r_M": get("M_off") * mu / P1, "s_Moff": get("var_M_off") * mu / P1 ** 2, "s_Mon": get("var_M_on") / P1 ** 2,
            "r_Q": get("Q_T") / get("Q_on"), "s_Qon": get("var_Q_on") / get("Q_on") ** 2,
            "s_QT": get("var_Q_T") / get("Q_on") ** 2, "r_G": get("G_T") / get("G_on"),
            "s_Gon": get("var_G_on") / get("G_on") ** 2, "s_GT": get("var_G_T") / get("G_on") ** 2,
            "s_GN": get("var_G_N") / get("G_on") ** 2, "d2": window["Phi2"] / P1 ** 2 - 1,
            "dPsi": window["Psi"] / P1 - 0.5, "rho": K / V, "Lambda": mu / (V + K)}


def normalized_terms_ell(pairs=None, as_written: bool = False) -> dict:
    """{name: (tags, expression)}: every term of `symbolic_terms_ell` (leading background,
    n = mu) in units of the signal per earlier a, h0 = (beta / L) G_on Q_on Phi1 (h0^2 for a
    covariance), written in the ratios of `RATIOS`, mu, ell, Lambda and rho = K / V
    (Lambda = p_T mu, so p_T = Lambda / mu, V = mu / (Lambda (1 + rho)), K = rho V)."""
    S, R, C = SYMBOLS, ORDER_SYMBOLS, COUNT_SYMBOLS
    mu, P1, Q_on, G_on, Lambda, rho = S["mu"], S["Phi1"], S["Q_on"], S["G_on"], S["Lambda"], C["rho"]
    V = mu / (Lambda * (1 + rho))
    substitution = {
        S["n"]: mu, S["V"]: V, S["K"]: rho * V,
        S["M_off"]: R["r_M"] * P1 / mu, S["var_M_off"]: R["s_Moff"] * P1 ** 2 / mu, S["var_M_on"]: R["s_Mon"] * P1 ** 2,
        S["Q_T"]: R["r_Q"] * Q_on, S["var_Q_on"]: R["s_Qon"] * Q_on ** 2, S["var_Q_T"]: R["s_QT"] * Q_on ** 2,
        S["G_T"]: R["r_G"] * G_on, S["var_G_on"]: R["s_Gon"] * G_on ** 2, S["var_G_T"]: R["s_GT"] * G_on ** 2,
        S["var_G_N"]: R["s_GN"] * G_on ** 2,
        S["Phi2"]: P1 ** 2 * (1 + R["d2"]), S["Psi"]: P1 * (sp.Rational(1, 2) + R["dPsi"]),
    }
    h0 = S["beta"] * G_on * Q_on * P1 / S["L"]
    allowed = {mu, S["ell"], Lambda, rho, S["s"], S["s2"], *R.values()}
    out = {}
    for name, (tags, expression) in symbolic_terms_ell(pairs, "leading", as_written).items():
        power = 1 if tags["channel"] == "mean" else 2
        value = sp.expand(sp.cancel(expression.subs(substitution) / h0 ** power))
        if not value.free_symbols <= allowed:
            raise AssertionError(f"{name}: left {value.free_symbols - allowed}")
        out[name] = (tags, value)
    return out


def monomials(expression: sp.Expr) -> dict:
    """{order: coefficient}: an expanded expression grouped by its powers of mu and of the
    ratios (`order` a tuple of (symbol name, power), sorted), each coefficient a function of
    ell, Lambda, rho and the classes s, s2 only."""
    order_names = {SYMBOLS["mu"], *ORDER_SYMBOLS.values()}
    groups = {}
    for term in sp.Add.make_args(sp.expand(expression)):
        if term == 0:
            continue
        key, coefficient = [], sp.Integer(1)
        # a factored product, so that mu in a denominator such as (mu rho + mu) is a factor
        for base, exponent in sp.expand_power_base(sp.factor(term), force=True).as_powers_dict().items():
            if base in order_names:
                key.append((base.name, int(exponent)))
            else:
                coefficient *= base ** exponent
        key = tuple(sorted(key))
        groups[key] = groups.get(key, sp.Integer(0)) + coefficient
    return {key: sp.factor(value) for key, value in groups.items() if sp.simplify(value) != 0}


__all__ = ["SYMBOLS", "SymbolicEll", "background_sums", "symbolic_terms_ell", "TRIGGER_PAIRS", "RATIOS",
           "S_SYMBOLS", "SymbolicS", "symbolic_terms_S",
           "ORDER_SYMBOLS", "COUNT_SYMBOLS", "ratio_values", "normalized_terms_ell", "monomials"]
