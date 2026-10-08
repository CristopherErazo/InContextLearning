"""Phase 7: the generated LaTeX of the final report (term inventory, power counting, presets),
from the symbolic layer and the Phase 4-6 tables, so that the report cannot drift from the
code. Writes data/report/{inventory_trigger,inventory_spread,inventory_nontrigger,orders,
minimal}.tex (longtable bodies and equations, for \\input or pasting).

    python -m scripts.report_tables
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import sympy as sp

from icl.theory.terms.presets import ADDED, DROPPED
from icl.theory.terms.symbolic import (COUNT_SYMBOLS, ORDER_SYMBOLS, S_SYMBOLS, SYMBOLS, monomials, normalized_terms_ell,
                                       symbolic_terms_ell, symbolic_terms_S)
from icl.theory.terms.trigger import _structure

OUT = Path("data/report")
HIERARCHY = Path("data/hierarchy")
PAIR_ORDER = ("on", "T", "b", "on,on", "T,T", "T,T'", "on,T", "b,b", "on,b", "b,T", "b,b'")
NONTRIGGER_ORDER = ("T,T", "T,T'", "rho,rho", "T,rho", "rho,rho'")
PAIR_TEX = {"on": r"$\E\hb^{on}$", "T": r"$\E\hb_\tau$", "b": r"$\E\hb_b$", "on,on": r"$\Var h^{on}$",
            "T,T": r"$\Var h_\tau$", "T,T'": r"$\Cov(h_\tau,h_{\tau'})$", "on,T": r"$\Cov(h^{on},h_\tau)$",
            "b,b": r"$\Var h_b$", "on,b": r"$\Cov(h^{on},h_b)$", "b,T": r"$\Cov(h_b,h_\tau)$",
            "b,b'": r"$\Cov(h_b,h_{b'})$", "rho,rho": r"$\Var h_\rho$", "T,rho": r"$\Cov(h_\tau,h_\rho)$",
            "rho,rho'": r"$\Cov(h_\rho,h_{\rho'})$"}

STAT_TEX = {"ell": r"\ell", "f": "f", "Nm": r"M^{on}\Nphi", "Fm": r"M^{on}\Fphi", "W": r"\Wc", "P": r"\Pc",
            "R": r"\Rc", "Nm2": r"(M^{on})^2\Ntwo", "Rm": r"M^{on}\Rc^{\varphi}", "X1": r"\mathcal X_1",
            "X2": r"\mathcal X_2", "pairs_phi": r"(\ell+f)(\ell+f-1)", "Fm2": r"(M^{on})^2\mathcal F^{\varphi^2}",
            "c": "c_{B}", "Ub": r"M^{on}\Uphi_{B}", "Utr": r"M^{on}\mathcal U^{\varphi,{\rm tr}}_{B}",
            "Wb": r"\Wbar_{B}", "Pb": r"\Pbar_{B}", "Yb": r"(M^{on})^2\mathcal Y_{B}", "pairs_b": r"c_{B}(c_{B}-1)",
            "UbN2": r"(M^{on})^2\mathcal U^{\varphi^2,{\rm N}}_{B}"}
PARAM_TEX = {"M_off": r"M^{\rm off}", "Q_on": "Q^{on}", "Q_T": r"Q^{\T}", "dQ": r"\Delta Q", "q2_T": r"q^2_{\T}",
             "dq2": r"\Delta q^2", "G_on": r"\Gamma^{on}", "G_T": r"\Gamma^{\T}", "var_M_on": r"(\sigma_M^{on})^2",
             "var_M_off": r"(\sigma_M^{\rm off})^2", "var_Q_on": r"(\sigma_Q^{on})^2", "var_Q_T": r"(\sigma_Q^{\T})^2",
             "var_Q_N": r"(\sigma_Q^{\rm N})^2", "var_G_on": r"(\sigma_\Gamma^{on})^2", "var_G_T": r"(\sigma_\Gamma^{\T})^2",
             "var_G_N": r"(\sigma_\Gamma^{\rm N})^2", "kappa_2": r"\kappa_2", "kappa_N": r"\kN", "kappa_NN": r"\kNN",
             "p_T": r"\pT", "kappa_bi": r"\kb", "D": r"\mathcal D", "G1": r"\mathcal G_1", "K1": r"\mathcal K_1",
             "K2": r"\mathcal K_2", "J2": r"\mathcal J_2", "Mall": r"M^{on}\Phihat", "n": "n",
             "nn": r"\tfrac{n(n-1)}2", "beta": r"\beta", "L": "L"}
RATIO_TEX = {"mu": r"\mu", "r_M": "r_M", "s_Moff": r"s_M^{\rm off}", "s_Mon": r"s_M^{on}", "r_Q": "r_Q",
             "s_Qon": r"s_Q^{on}", "s_QT": r"s_Q^{\T}", "r_G": r"r_\Gamma", "s_Gon": r"s_\Gamma^{on}",
             "s_GT": r"s_\Gamma^{\T}", "s_GN": r"s_\Gamma^{\rm N}", "d2": r"\delta_2", "dPsi": r"\hat\delta_\Psi"}


SORT = {**dict.fromkeys(("G_on", "G_T", "var_G_on", "var_G_T", "var_G_N"), "a"),
        **dict.fromkeys(("Q_on", "Q_T", "dQ", "q2_T", "dq2", "var_Q_on", "var_Q_T", "var_Q_N"), "b"),
        **dict.fromkeys(("kappa_2", "kappa_N", "kappa_NN", "p_T", "kappa_bi"), "c"),
        **dict.fromkeys(("M_off", "var_M_on", "var_M_off"), "d"),
        **dict.fromkeys(("D", "G1", "K1", "K2", "J2", "Mall", "n", "nn"), "e")}


def printable(expression: sp.Expr, column: str = "b") -> tuple[sp.Expr, dict]:
    """The expression on printing symbols whose names sort as the appendix writes a product
    (readout, Q, rates, M, sums, statistics), and their LaTeX names (the b column `column`)."""
    replace, names = {}, {}
    for name, symbol in S_SYMBOLS.items():
        base = name[:-2] if name.endswith("_2") and name[:-2] in STAT_TEX else name
        if base in STAT_TEX:
            tex = STAT_TEX[base].replace("{B}", "{" + column + ("'" if base != name else "") + "}")
            key = "f" + name
        else:
            tex, key = PARAM_TEX[name], SORT.get(name, "z") + name
        printing = sp.Symbol(key, real=True)
        replace[symbol], names[printing] = printing, tex
    return expression.xreplace(replace), names


def breakable(tex: str) -> str:
    """Plain parentheses, so that TeX can break a long inline formula inside them."""
    return tex.replace(r"\left(", "(").replace(r"\right)", ")")


def formula(expression: sp.Expr, power: int, column: str = "b") -> tuple[str, str]:
    """(sign, LaTeX) of a term as (beta/L)^power times the rest."""
    prefactor = (S_SYMBOLS["beta"] / S_SYMBOLS["L"]) ** power
    rest = sp.factor_terms(expression / prefactor)
    sign = "+"
    if rest.as_coeff_Mul()[0] < 0:
        sign, rest = "-", -rest
    rest, names = printable(rest, column)
    tex = breakable(sp.latex(rest, symbol_names=names, long_frac_ratio=4))
    if not (rest.is_Atom or rest.is_Mul or rest.is_Pow):
        tex = r"ig(" + tex + r"ig)"
    return sign, (r"\pref" if power == 2 else r"\tfrac\beta L") + r"\," + tex


def short(name: str) -> str:
    text = name.split("/", 1)[1].replace("_", r"\_").replace(".", r".\allowbreak{}").replace("/", r"/\allowbreak{}")
    return r"\code{" + text + "}"


def tex_name(preset: str) -> str:
    return preset.replace("_", r"\_")


def first_preset(name: str) -> str:
    for preset, names in ADDED.items():
        if name in names:
            return preset
    return "kept"


def inventory(query: str) -> str:
    terms = symbolic_terms_S(query)
    column = "b" if query == "trigger" else r"\rho"
    order = PAIR_ORDER if query == "trigger" else NONTRIGGER_ORDER
    rows = []
    for pair in order:
        members = [(name, tags, expr) for name, (tags, expr) in terms.items() if tags["pair"] == pair]
        if not members:
            continue
        rows.append(r"\multicolumn{3}{l}{" + PAIR_TEX[pair] + r"}\\*")
        for name, tags, expr in members:
            power = 1 if tags["channel"] == "mean" else 2
            preset = first_preset(name) if query == "trigger" else ""
            sign, tex = formula(expr, power, column)
            rows.append(f"{short(name)} & ${'-' if sign == '-' else ''}{tex}$ & {tex_name(preset)}\\\\")
        rows.append(r"\addlinespace")
    return "\n".join(rows)


def spread_inventory() -> str:
    """The spread terms given ell: (beta/L)^2 Gamma_X Gamma_Y c_i c_j Cov(S_i, S_j | ell)."""
    S = S_SYMBOLS
    p = {name: S[name] for name in ("M_off", "Q_on", "Q_T", "dQ", "q2_T", "dq2", "G_on", "G_T", "var_M_on",
                                    "var_M_off", "var_Q_on", "var_Q_T", "var_Q_N")}
    zero = {name: sp.Integer(0) for name in ("D", "G1", "K1", "K2", "J2", "Mall", "n", "nn")}
    rates = {name: sp.Integer(0) for name in ("kappa_2", "kappa_N", "kappa_NN", "p_T", "kappa_bi")}
    means = _structure(p, zero, rates)["means"]
    gamma = {"on": S["G_on"], "T": S["G_T"], "b": S["G_on"]}
    stat_tex = lambda form, prime="": " + ".join(
        (f"{sp.latex(c)}\\," if c != 1 else "") + STAT_TEX[n].replace("{B}", "{b" + prime + "}") for n, c in form.coefs.items())
    ell_terms = symbolic_terms_ell()
    rows = []
    for X, Y, pair in (("on", "on", "on,on"), ("T", "T", "T,T"), ("T", "T", "T,T'"), ("on", "T", "on,T"),
                       ("b", "b", "b,b"), ("on", "b", "on,b"), ("b", "T", "b,T"), ("b", "b", "b,b'")):
        rows.append(r"\multicolumn{3}{l}{" + PAIR_TEX[pair] + r"}\\*")
        for i, (mech_i, _, coef_i, stat_i) in enumerate(means[X]):
            for j, (mech_j, _, coef_j, stat_j) in enumerate(means[Y]):
                name = f"{pair}/spread/{mech_i}*{mech_j}"
                if name not in ell_terms:
                    continue
                twice = 2 if X == Y and j > i else 1
                prime = "'" if pair == "b,b'" else ""
                coefficient, names = printable(twice * gamma[X] * gamma[Y] * coef_i * coef_j)
                tex = (rf"\pref\,{sp.latex(coefficient, symbol_names=names)}\,"
                       rf"\Cov\!\big({stat_tex(stat_i)},{stat_tex(stat_j, prime)}\,\big|\,\ell\big)")
                rows.append(f"{short(name)} & ${tex}$ & {tex_name(first_preset(name))}\\\\")
        rows.append(r"\addlinespace")
    return "\n".join(rows)


def orders() -> str:
    """Per term: the leading monomial at run_001 (3000, mu = 512, ell = 1) with its coefficient,
    E^c, the Phase 4 share at 3000, the local exponent and the preset that drops it."""
    normalized = normalized_terms_ell()
    phase5 = pd.read_csv(HIERARCHY / "phase5_terms.csv").set_index("term")
    names = {**{s: RATIO_TEX[k] for k, s in ORDER_SYMBOLS.items()}, SYMBOLS["mu"]: r"\mu", SYMBOLS["ell"]: r"\ell",
             SYMBOLS["Lambda"]: r"\Lambda", COUNT_SYMBOLS["rho"]: r"\rho"}
    rows = []
    for pair in PAIR_ORDER:
        rows.append(r"\multicolumn{6}{l}{" + PAIR_TEX[pair] + r"}\\*")
        for name, (tags, expression) in normalized.items():
            if tags["pair"] != pair:
                continue
            row = phase5.loc[name]
            key_text = row["3000/estimate_monomial"]
            key = tuple(sorted((part.split("^")[0], int(part.split("^")[1]) if "^" in part else 1)
                               for part in key_text.split())) if key_text != "1" else ()
            coefficient = monomials(expression).get(key, sp.Integer(1))
            monomial = sp.Mul(*(sp.Symbol(n, real=True) ** k for n, k in key))
            monomial_tex = sp.latex(monomial, symbol_names={sp.Symbol(k, real=True): v for k, v in RATIO_TEX.items()})
            coefficient_tex = sp.latex(sp.factor(coefficient), symbol_names=names)
            if not key:
                lead = sp.latex(sp.factor(coefficient), symbol_names=names)
            elif coefficient == 1:
                lead = monomial_tex
            else:
                lead = (rf"\big({coefficient_tex}\big)\,{monomial_tex}"
                                            if coefficient.is_Add else rf"{coefficient_tex}\,{monomial_tex}")
            rows.append(f"{short(name)} & ${lead}$ & {_number(row['3000/estimate_counts'])} & "
                        f"{_number(row['3000/max_share'])} & ${row['robust/relative_local_exponent']:+.1f}$ & "
                        f"{tex_name(first_preset(name))}\\\\")
        rows.append(r"\addlinespace")
    return "\n".join(rows)


def _number(x: float) -> str:
    if x == 0 or x != x:
        return "0"
    exponent = int(sp.floor(sp.log(abs(x), 10)))
    return f"${x / 10 ** exponent:.1f}\\times10^{{{exponent}}}$" if exponent < -1 else f"${x:.2g}$"


def minimal() -> str:
    """The minimal preset given S: per pair, the kept terms as one sum (aligned)."""
    terms = symbolic_terms_S("trigger")
    blocks = []
    for pair in PAIR_ORDER:
        kept = [(name, tags, expr) for name, (tags, expr) in terms.items()
                if tags["pair"] == pair and name not in DROPPED["minimal"]]
        if not kept:
            continue
        power = 1 if kept[0][1]["channel"] == "mean" else 2
        lines = []
        for k, (_, _, expr) in enumerate(kept):
            sign, tex = formula(expr, power)
            lines.append("&" + ("=" + ("-" if sign == "-" else "") if k == 0 else sign) + tex)
        lhs = PAIR_TEX[pair].strip("$") + r"\,\big|_{\bm{\mathcal S}}"
        blocks.append(r"\begin{align*}" + "\n" + lhs + "\n" + "\\\\\n".join(lines) + "\n" + r"\end{align*}")
    return "\n".join(blocks)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inventory_trigger.tex").write_text(inventory("trigger") + "\n")
    (OUT / "inventory_nontrigger.tex").write_text(inventory("nontrigger") + "\n")
    (OUT / "inventory_spread.tex").write_text(spread_inventory() + "\n")
    (OUT / "orders.tex").write_text(orders() + "\n")
    (OUT / "minimal.tex").write_text(minimal() + "\n")
    print(f"wrote {sorted(p.name for p in OUT.glob('*.tex'))}")


if __name__ == "__main__":
    main()
