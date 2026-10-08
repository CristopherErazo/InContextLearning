"""The sympy terms at a trigger query given ell (icl.theory.terms.symbolic) against the
tensor ones (trigger_terms_ell), term by term, with a profile and every variance."""
from __future__ import annotations

import math

import pytest
import sympy as sp
import torch

from icl.theory.terms import Bilinear, trigger_terms_ell, window_moments
from icl.theory.terms.symbolic import SYMBOLS, symbolic_terms_ell

V, L, K, BETA = 48, 96, 9, 0.25
OP = {"M_on": 6.0, "M_off": 0.03, "Q_on": 9.0, "Q_T": -0.7, "G_on": 5.0, "G_T": -0.8,
      "var_M_on": 0.4, "var_M_off": 0.05, "var_Q_on": 1.3, "var_Q_T": 0.6, "var_Q_N": 0.2,
      "var_G_on": 0.5, "var_G_T": 0.3, "var_G_N": 0.2,
      "M_profile": 6.0 * (1 + 0.3 * torch.sin(torch.arange(L - 1, dtype=torch.float64) / 9))}
MU = torch.tensor([12, 40, 40, 96, 96])
ELL = torch.tensor([0, 1, 3, 2, 6])


@pytest.fixture(scope="module")
def symbolic():
    return symbolic_terms_ell()


def _numbers(Lambda: str) -> list[dict]:
    window = window_moments(OP, MU, L)
    n = (MU - 2).clamp(min=0).double()
    p_T = 1 / (V + K)
    lam = p_T * MU.double() if Lambda == "mu" else p_T * (n - 2 * ELL).clamp(min=0)
    fixed = {name: OP[name] for name in ("M_off", "Q_on", "Q_T", "G_on", "G_T", "var_M_on", "var_M_off", "var_Q_on",
                                         "var_Q_T", "var_G_on", "var_G_T", "var_G_N")}
    fixed |= {"beta": BETA, "L": L, "V": V, "K": K}
    return [fixed | {"mu": float(MU[i]), "n": float(n[i]), "ell": float(ELL[i]), "Lambda": float(lam[i]),
                     **{name: float(window[name][i]) for name in ("Phi1", "Phi2", "Psi")}} for i in range(len(MU))]


def _numeric(value, nb: int):
    """{(s, s2): (rows,)} of a term: per row; per class (column 0 an output, the last a plain
    token); per pair of classes of two distinct columns."""
    if isinstance(value, Bilinear):
        dense = value.dense()
        return {(1, 1): dense[:, 0, 1], (1, 0): dense[:, 0, nb - 1], (0, 1): dense[:, nb - 1, 0],
                (0, 0): dense[:, nb - 2, nb - 1]}
    value = torch.as_tensor(value, dtype=torch.float64)
    if value.dim() == 2:
        return {(1, None): value[:, 0], (0, None): value[:, nb - 1]}
    return {(None, None): value.expand(len(MU))}


@pytest.mark.parametrize("Lambda", ["mu", "n-2ell"])
def test_symbolic_terms_equal_the_tensor_terms(symbolic, Lambda):
    table = trigger_terms_ell(MU, ELL, OP, BETA, L, V, K, Lambda=Lambda, background="exact")
    nb = V - K - 1
    assert set(symbolic) == {term.name for term in table}
    rows = _numbers(Lambda)
    names = [SYMBOLS[name] for name in rows[0]] + [SYMBOLS["s"], SYMBOLS["s2"]]
    for term in table:
        tags, expression = symbolic[term.name]
        assert tags["pair"] == term.pair and tags["channel"] == term.channel and tags["mechanism"] == term.mechanism
        function = sp.lambdify(names, expression, "math")
        for (s, s2), values in _numeric(term.value, nb).items():
            for row, numbers in enumerate(rows):
                got = function(*numbers.values(), s if s is not None else 0, s2 if s2 is not None else 0)
                want = values[row].item()
                assert got == pytest.approx(want, rel=1e-10, abs=1e-14 * (1 + abs(want))), (term.name, s, s2, row)


def test_leading_background_is_the_leading_order_of_the_exact_one():
    from icl.theory.terms.symbolic import background_sums
    exact, leading = background_sums(SYMBOLS, "exact"), background_sums(SYMBOLS, "leading")
    n, mu = SYMBOLS["n"], SYMBOLS["mu"]
    for name in ("D", "G1", "K1", "K2", "J2"):
        x = sp.Symbol("x", positive=True)              # n = mu = 1/x, x -> 0
        ratio = sp.limit(sp.simplify((exact[name] / leading[name]).subs({n: 1 / x, mu: 1 / x})), x, 0)
        assert ratio == 1, name
    assert not math.isnan(float(exact["nn"].subs(n, 5)))


def test_normalized_terms_are_the_terms_in_units_of_the_signal():
    """normalized_terms_ell x h0^k = symbolic_terms_ell (leading background, n = mu) at the
    ratio values of the order parameters; and the monomials add up to the expression."""
    from icl.theory.terms.symbolic import (COUNT_SYMBOLS, ORDER_SYMBOLS, monomials, normalized_terms_ell,
                                           ratio_values)
    op = {name: value for name, value in OP.items() if name != "M_profile"}
    op["M_on"] = 6.0
    mu, ell = 40, 3
    window = {name: float(value) for name, value in window_moments(op, torch.tensor([mu]), L).items()}
    ratios = ratio_values(op, window, mu, V, K)
    S = SYMBOLS
    raw_numbers = {S[name]: op[name] for name in ("M_off", "Q_on", "Q_T", "G_on", "G_T", "var_M_on", "var_M_off",
                                                   "var_Q_on", "var_Q_T", "var_G_on", "var_G_T", "var_G_N")}
    raw_numbers |= {S["beta"]: BETA, S["L"]: L, S["V"]: V, S["K"]: K, S["mu"]: mu, S["n"]: mu, S["ell"]: ell,
                    S["Lambda"]: ratios["Lambda"], S["s"]: 1, S["s2"]: 0, **{S[name]: window[name] for name in window}}
    numbers = {symbol: ratios[name] for name, symbol in ORDER_SYMBOLS.items()}
    numbers |= {COUNT_SYMBOLS["rho"]: ratios["rho"], S["Lambda"]: ratios["Lambda"], S["mu"]: mu, S["ell"]: ell,
                S["s"]: 1, S["s2"]: 0}
    h0 = BETA * op["G_on"] * op["Q_on"] * window["Phi1"] / L
    raw = symbolic_terms_ell(background="leading")
    pairs = ["on", "T", "b", "on,on", "T,T", "on,T", "b,b", "on,b", "b,T", "b,b'"]
    normalized = normalized_terms_ell(pairs)
    for name in ("on/mean/signal", "T/mean/bulk_a", "on,on/attention/C_phi.incoherent", "on,on/spread/signal*signal",
                 "T,T/attention/C_all.token_coherent.BB", "b,b'/attention/C_bb'.token_coherent.NB",
                 "on,b/attention/C_phi_b.a_coherent", "b,T/attention/C_b_all.token_coherent.tr_all"):
        tags, expression = normalized[name]
        power = 1 if tags["channel"] == "mean" else 2
        want = float(raw[name][1].subs(raw_numbers))
        assert float(expression.subs(numbers)) * h0 ** power == pytest.approx(want, rel=1e-9, abs=1e-15), name
        total = sum(coefficient * sp.Mul(*(sp.Symbol(symbol, real=True) ** power for symbol, power in key))
                    for key, coefficient in monomials(expression).items())
        assert float(sp.expand(total - expression).subs(numbers)) == pytest.approx(0, abs=1e-9 * (1 + abs(want))), name


@pytest.mark.parametrize("query", ["trigger", "nontrigger"])
def test_symbolic_terms_given_S_equal_the_tensor_terms(query):
    """symbolic_terms_S evaluated on the statistics of measured rows = trigger_terms /
    nontrigger_terms, term by term (one output and one plain column; b-b' pairs)."""
    from icl import generate_icl_batch, measure_nontrigger_variables, measure_variables
    from icl.theory.sums import _diagonal
    from icl.theory.terms import nontrigger_statistics, nontrigger_terms, trigger_statistics, trigger_terms
    from icl.theory.terms.forms import B_STATS
    from icl.theory.terms.symbolic import S_SYMBOLS, symbolic_terms_S
    from icl.theory.terms.trigger import _background, _derived, _parameters, coincidence_rates
    batch = generate_icl_batch(64, V, L, K, generator=torch.Generator().manual_seed(3))
    if query == "trigger":
        variables = measure_variables(batch, [12, 40, 96])
        table, stats = trigger_terms(variables, OP, BETA, L), trigger_statistics(variables, OP, L)
        nb, outputs = V - K - 1, K - 1
    else:
        variables = measure_nontrigger_variables(batch, [12, 40, 96])
        table, stats = nontrigger_terms(variables, OP, BETA, L), nontrigger_statistics(variables, OP, L)
        nb, outputs = V - K, K
    rows = min(len(variables["mu"]), 12)
    p = _derived(_parameters(OP, torch.float64, None))
    diag = _diagonal(OP, L, torch.float64, None)
    bg = _background(diag, p, (variables["mu"] - 2).clamp(min=0), torch.float64)
    numbers = {name: float(p[name]) for name in ("M_off", "Q_on", "Q_T", "dQ", "q2_T", "dq2", "G_on", "G_T", "var_M_on",
                                                 "var_M_off", "var_Q_on", "var_Q_T", "var_Q_N", "var_G_on", "var_G_T",
                                                 "var_G_N")}
    numbers |= {name: float(value) for name, value in coincidence_rates(V, K).items()} | {"beta": BETA, "L": L}
    symbolic = symbolic_terms_S(query)
    assert set(symbolic) == {term.name for term in table}
    columns = {"out": 0, "plain": nb - 1}
    for term in table:
        expression = symbolic[term.name][1]
        names = sorted(expression.free_symbols, key=lambda symbol: symbol.name)
        function = sp.lambdify(names, expression, "math")
        value = term.value
        for row in range(rows):
            per_row = {name: (float(bg[name][row]) if torch.is_tensor(bg[name]) and bg[name].dim() else float(bg[name]))
                       for name in ("D", "G1", "K1", "K2", "J2", "Mall", "n", "nn")}
            if isinstance(value, Bilinear):
                cases = [((0, nb - 1), value.dense()[row, 0, nb - 1].item())]
            elif torch.is_tensor(value) and value.dim() == 2:
                cases = [((column, None), value[row, column].item()) for column in columns.values()]
            else:
                cases = [((None, None), torch.as_tensor(value).expand(len(variables["mu"]))[row].item())]
            for (b, b2), want in cases:
                env = numbers | per_row
                for name, stat in stats.items():
                    if stat.dim() == 1:
                        env[name] = float(stat[row])
                    elif b is not None:
                        env[name] = float(stat[row, b])
                        if b2 is not None and name in B_STATS:
                            env[f"{name}_2"] = float(stat[row, b2])
                got = function(*(env[symbol.name] for symbol in names))
                assert got == pytest.approx(want, rel=1e-9, abs=1e-14 * (1 + abs(want))), (term.name, row, b)
