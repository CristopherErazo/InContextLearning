"""The terms at a non-trigger query, given S_mu (`nontrigger_terms`, eq. pairs_nontrigger)
and given mu (`nontrigger_terms_mu`, eqs. nontrigger_means, pairs_nontrigger_mean). The
mean logits vanish; the noise is the background of eq. background_N with sigma_Q^N, with
the corrected token-coherent rule. Classes: the K triggers "T" and the V-K non-triggers
"rho" (the K outputs, then the plain tokens)."""
from __future__ import annotations

import torch

import math

from ..ansatz import _take
from ..sums import _diagonal
from ..query_table import QueryTable
from .forms import Form, GivenS, Piece
from .table import Term, TermTable
from .trigger import (_background, _ell_setup, _evaluate, _expand, _predecessor_sums, _readout_factors, _tensors,
                      coincidence_rates)

S_LABEL, MU_LABEL = "eq-apx:pairs_nontrigger", "eq-apx:nontrigger_means"


KEYSETS_OF_PAIR = {"T,T": ("C_all", "E"), "T,T'": ("C_all",), "rho,rho": ("C_rho", "E"), "T,rho": ("C_rho_all",),
                   "rho,rho'": ("C_rhorho'",)}


def _needed(pairs):
    if pairs is None:
        return None
    unknown = set(pairs) - set(KEYSETS_OF_PAIR)
    if unknown:
        raise ValueError(f"unknown pairs {sorted(unknown)}; pairs: {tuple(KEYSETS_OF_PAIR)}")
    return {keyset for pair in pairs for keyset in KEYSETS_OF_PAIR[pair]}


def _structure(p: dict, bg: dict, rates: dict, as_written: bool = False) -> dict:
    """The key-set sums and the energy as pieces; as_written: as `trigger._structure`."""
    S = Form.of
    k2, kN, kNN, pT = rates["kappa_2"], rates["kappa_N"], rates["kappa_NN"], rates["p_T"]
    sN2, M_off = p["var_Q_N"], p["M_off"]
    D, G1, K1, K2, J2, Mall, n, nn = (bg[name] for name in ("D", "G1", "K1", "K2", "J2", "Mall", "n", "nn"))
    one, c = Form(const=1.0), S("c")
    gN, gtr, gB = S("Ub") - S("Utr"), S("Utr"), M_off * S("Wb")
    bigram = Piece("token_coherent", "bigram", ("var_Q_N", "Y_rho"), sN2, S("Yb"))

    def tc_cross(cross):
        return [Piece("token_coherent", "BB", ("var_Q_N", "kappa_2", "gB", "gB"), sN2 * k2, gB, gB, cross),
                Piece("token_coherent", "NB", ("var_Q_N", "kappa_N", "gN", "gB"), sN2 * kN, gN, gB, cross),
                Piece("token_coherent", "BN", ("var_Q_N", "kappa_N", "gB", "gN"), sN2 * kN, gB, gN, cross),
                Piece("token_coherent", "TB", ("var_Q_N", "p_T", "gtr", "gB"), sN2 * pT, gtr, gB, cross),
                Piece("token_coherent", "BT", ("var_Q_N", "p_T", "gB", "gtr"), sN2 * pT, gB, gtr, cross),
                Piece("token_coherent", "NN", ("var_Q_N", "kappa_NN", "gN", "gN"), sN2 * kNN, gN, gN, cross)]

    keysets = {
        "C_all": [Piece("incoherent", "", ("var_Q_N", "D"), sN2 * D, one),
                  Piece("source_coherent", "", ("var_Q_N", "J2"), sN2 * J2, one),
                  Piece("token_coherent", "BB", ("var_Q_N", "kappa_2", "K1^2-J2"),
                        sN2 * k2 * (K1 ** 2 - (0.0 if as_written else J2)), one)],
        "C_rho": [Piece("incoherent", "", ("var_Q_N", "D"), sN2 * D / n, c),
                  Piece("source_coherent", "same_key", ("var_Q_N", "G1"), sN2 * G1 / n, c),
                  Piece("source_coherent", "diff_key", ("var_Q_N", "J2-G1"), sN2 * (J2 - G1) / n ** 2, S("pairs_b")),
                  Piece("token_coherent", "BB", ("var_Q_N", "kappa_2", "gB", "gB"), sN2 * k2, gB, gB),
                  Piece("token_coherent", "NB", ("var_Q_N", "kappa_N", "gN", "gB"), 2 * sN2 * kN, gN, gB),
                  Piece("token_coherent", "TB", ("var_Q_N", "p_T", "gtr", "gB"), 2 * sN2 * pT, gtr, gB),
                  Piece("token_coherent", "NN", ("var_Q_N", "kappa_NN", "gN", "gN"), sN2 * kNN, gN, gN,
                        linear=None if as_written else -S("UbN2")),
                  bigram],
        "C_rho_all": [Piece("incoherent", "", ("var_Q_N", "D"), sN2 * D / n, c),
                      Piece("source_coherent", "share", ("var_Q_N", "J2"), sN2 * J2 / n, c),
                      Piece("token_coherent", "B_all", ("var_Q_N", "kappa_2", "gB", "K1"), sN2 * k2 * K1, gB),
                      Piece("token_coherent", "N_all", ("var_Q_N", "kappa_N", "gN", "K1"), sN2 * kN * K1, gN),
                      Piece("token_coherent", "tr_all", ("var_Q_N", "p_T", "gtr", "gB_all"), sN2 * pT * M_off * nn,
                            gtr),
                      bigram],
        "C_rhorho'": [Piece("source_coherent", "diff_key", ("var_Q_N", "J2-G1"), sN2 * (J2 - G1) / n ** 2, c, c, True),
                      *tc_cross(True)],
    }
    same_key, diff_key = (0.0, 0.0) if as_written else (G1, J2 - G1)
    energy = [Piece("incoherent", "same_key", ("var_Q_N", "D"), sN2 * D, one),
              Piece("source_coherent", "same_key", ("var_Q_N", "G1"), sN2 * G1, one),
              Piece("token_coherent", "same_key", ("var_Q_N", "kappa_2", "K2-G1"), sN2 * k2 * (K2 - same_key), one),
              Piece("source_coherent", "diff_key", ("kappa_2", "var_Q_N", "J2-G1"), k2 * sN2 * (J2 - G1), one),
              Piece("token_coherent", "diff_key", ("kappa_2^2", "var_Q_N", "K1^2-K2-J2+G1"),
                    k2 ** 2 * sN2 * (K1 ** 2 - K2 - diff_key), one),
              Piece("token_coherent", "bigram", ("var_Q_N", "kappa_bi-kappa_2^2", "Mall^2"),
                    sN2 * (rates["kappa_bi"] - k2 ** 2) * Mall ** 2, one)]
    return {"keysets": keysets, "energy": energy, "own": {}}


def assemble_nontrigger(evaluated: dict, factors: dict, label: str, nb: int, pairs=None) -> list[Term]:
    terms = []

    def add(pair, channel, column, keyset, piece, factor_name, value, sign=1.0):
        if pairs is not None and pair not in pairs:
            return
        detail = piece.detail
        name = f"{pair}/{channel}{'.' + column if column else ''}/{keyset}.{piece.mechanism}{'.' + detail if detail else ''}"
        terms.append(Term(name=name, pair=pair, channel=channel, column=column, mechanism=piece.mechanism,
                          keyset=keyset, detail=detail, label=label, params=(factor_name, *piece.params),
                          value=factors[factor_name] * sign * value))

    keysets, energy = evaluated["keysets"], evaluated["energy"]
    keysets = {name: keysets.get(name, []) for name in ("C_all", "C_rho", "C_rho_all", "C_rhorho'")}
    for piece, value in keysets["C_all"]:
        add("T,T", "attention", "", "C_all", piece, "G_T^2", value)
    for piece, value in energy:
        add("T,T", "readout", "all", "E", piece, "var_G_T", value)
    for piece, value in keysets["C_all"]:
        add("T,T'", "attention", "", "C_all", piece, "G_T^2", value)
    for piece, value in keysets["C_rho"]:
        add("rho,rho", "attention", "", "C_rho", piece, "G_on^2", value)
    for piece, value in keysets["C_rho"]:
        add("rho,rho", "readout", "own", "C_rho", piece, "var_G_on", value)
    for piece, value in energy:
        add("rho,rho", "readout", "other", "E", piece, "var_G_N", _expand(value, nb))
    for piece, value in keysets["C_rho"]:
        add("rho,rho", "readout", "other", "C_rho", piece, "var_G_N", value, sign=-1.0)
    for piece, value in keysets["C_rho_all"]:
        add("T,rho", "attention", "", "C_rho_all", piece, "G_on*G_T", value)
    for piece, value in keysets["C_rhorho'"]:
        add("rho,rho'", "attention", "", "C_rhorho'", piece, "G_on^2", value)
    return terms


def nontrigger_statistics(variables: QueryTable, order_params: dict, L: int) -> dict:
    """The per-token statistics of `forms` at the rows of a non-trigger table
    (`measure_nontrigger_variables` or `sample_nontrigger_variables`); without
    "U_triggered", the predecessor sums at their mean given the keys (`trigger_statistics`)."""
    dtype, device = variables["U_bar"].dtype, variables["U_bar"].device
    diag = _diagonal(order_params, L, dtype, device)
    m_tokens = _take(diag, variables["U_keys"])
    c = variables["U_bar"]
    Utr, Yb, UbN2 = _predecessor_sums(variables, m_tokens, int(variables["K"]))
    return {"ell": torch.zeros(len(c), dtype=dtype, device=device), "c": c, "Ub": m_tokens.sum(-1), "Utr": Utr,
            "Wb": variables["W_bar"], "Pb": torch.zeros_like(c), "Yb": Yb, "pairs_b": c * (c - 1), "UbN2": UbN2}


def nontrigger_terms(variables: QueryTable, order_params: dict, beta: float, L: int, pairs=None,
                     as_written: bool = False) -> TermTable:
    """Every term of the logit covariance at the non-trigger queries of `variables`
    (eq. pairs_nontrigger), given S_mu. Pairs "T,T", "T,T'", "rho,rho", "T,rho", "rho,rho'";
    pairs, as_written: as `trigger_terms`."""
    dtype, device = variables["U_bar"].dtype, variables["U_bar"].device
    K, V = int(variables["K"]), int(variables["V"])
    nb = V - K
    p = _tensors(order_params, dtype, device)
    diag = _diagonal(order_params, L, dtype, device)
    structure = _structure(p, _background(diag, p, (variables["mu"] - 2).clamp(min=0), dtype), coincidence_rates(V, K),
                           as_written)
    evaluated = _evaluate(structure, GivenS(nontrigger_statistics(variables, order_params, L), nb), _needed(pairs))
    terms = assemble_nontrigger(evaluated, _readout_factors(p, (beta / L) ** 2), S_LABEL, nb, pairs)
    return TermTable(terms, QueryTable({"mu": variables["mu"], "ell": variables["ell"], "K": K}), "S", "nontrigger",
                     K, V)


def nontrigger_terms_mu(mu, order_params: dict, beta: float, L: int, V: int, K: int, Lambda: str = "mu",
                        background: str = "exact", dtype=torch.float64, device=None, pairs=None,
                        as_written: bool = False) -> TermTable:
    """The terms of eq. pairs_nontrigger averaged over the sequence at a non-trigger query
    at mu (int or 1-D; eqs. nontrigger_means, pairs_nontrigger_mean): each term is the mean
    of the term of the same name of `nontrigger_terms` under A1-A3 (the moments of a
    non-target token of the trigger query at ell = 0, r_rho = 2 Lambda for the outputs,
    Lambda for the rest). Lambda, background: as `trigger_terms_ell` ("n-2ell" is
    p_T (mu - 2) here, the number of keys); pairs, as_written: as `trigger_terms`."""
    nb = V - K
    mu, _, p, bg, evaluator = _ell_setup(mu, 0, order_params, L, V, K, nb, K, Lambda, background, dtype, device)
    structure = _structure(p, bg, coincidence_rates(V, K), as_written)
    terms = assemble_nontrigger(_evaluate(structure, evaluator, _needed(pairs)), _readout_factors(p, (beta / L) ** 2),
                                MU_LABEL, nb, pairs)
    return TermTable(terms, QueryTable({"mu": mu, "ell": torch.zeros_like(mu), "K": K}), "mu", "nontrigger", K, V)


def nontrigger_second_order(table: TermTable, keep=None) -> torch.Tensor:
    """L_N - log V to second order in the logit fluctuation, per row of a non-trigger table:
    E[log (1/V) sum_c e^{xi_c - xi_bar}] = (1/2) E[(1/V) sum_c (xi_c - xi_bar)^2] + O(xi^3)
    = (1/2) [(1/V) sum_c Var xi_c - Var xi_bar], with the K triggers and the V - K tokens rho:

        sum_c Var xi_c = K Var_T + sum_rho Var_rho,
        V^2 Var xi_bar = K Var_T + K (K - 1) Cov_TT' + 2 K sum_rho Cov_T,rho + sum_rho Var_rho
                         + sum_{rho != rho'} Cov_rho,rho'.

    keep prunes the terms (as `TermTable.total`)."""
    K, V = table.K, table.V
    var_T, cov_T = table.total("T,T", keep), table.total("T,T'", keep)
    var_rho = table.total("rho,rho", keep).sum(1)
    cov_T_rho = table.total("T,rho", keep).sum(1)
    pairs = table.total("rho,rho'", keep)
    cov_rho = pairs.class_sums(table.b_classes.to(var_T.device)).sum((1, 2)) if pairs.parts else 0.0
    trace = K * var_T + var_rho
    total = K * var_T + K * (K - 1) * cov_T + 2 * K * cov_T_rho + var_rho + cov_rho
    return 0.5 * (trace / V - total / V ** 2)
