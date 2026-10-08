"""The terms at a trigger query, given S_mu (`trigger_terms`) and given ell
(`trigger_terms_ell`): eqs. mean_logits, keyset_sums, E, pairs_trigger of the appendix and
their averages, eqs. mean_logits_ell, var_logits, cov_logits, keyset_means, E_mean,
pairs_trigger_ell, with the corrected token-coherent rule (eqs. predecessor - tc_all,
g_sets).

The pieces are defined once (`_structure`), as products of linear forms in the
statistics; `GivenS` evaluates them on a table of variables, `GivenEll` replaces them by
their means given ell. `_assemble` then reads every pair of logits through the readout
(pair table of the subsection "Trigger query").
"""
from __future__ import annotations

import torch

from ..ansatz import _take
from ..sums import _diagonal, _key_sums, _parameters, _witness_sums
from ..query_table import QueryTable
from .forms import Form, GivenEll, GivenS, Piece
from .table import Term, TermTable

S_LABELS = {"mean": "eq-apx:mean_logits", "keyset": "eq-apx:keyset_sums", "energy": "eq-apx:E"}
ELL_LABELS = {"mean": "eq-apx:mean_logits_ell", "keyset": "eq-apx:keyset_means", "energy": "eq-apx:E_mean",
              "var": "eq-apx:var_logits", "cov": "eq-apx:cov_logits"}
LAMBDAS = ("mu", "n-2ell")
BACKGROUNDS = ("exact", "leading")


def coincidence_rates(V: int, K: int) -> dict[str, float]:
    """kappa_2, kappa_N, kappa_NN, p_T (eq. rates) and kappa_bi (R1)."""
    p_T = 1 / (V + K)
    return {"kappa_2": (V + 3 * K) * p_T ** 2, "kappa_N": (V + 2 * K) * p_T / V, "kappa_NN": (V + 2 * K) / V ** 2,
            "p_T": p_T, "kappa_bi": (K + 1 + 2 * K / V) * p_T ** 2}


def _derived(p: dict) -> dict:
    p = dict(p)
    p["dQ"] = p["Q_on"] - p["Q_T"]
    p["q2_T"] = p["Q_T"] ** 2 + p["var_Q_T"]
    p["q2_on"] = p["Q_on"] ** 2 + p["var_Q_on"]
    p["dq2"] = p["q2_on"] - p["q2_T"]
    return p


def _structure(p: dict, bg: dict, rates: dict, as_written: bool = False) -> dict:
    """The mean terms, the brackets, the seven key-set sums and the energy as pieces.
    p: order parameters (tensors), bg: the deterministic sums D, G1, K1, K2, J2, Mall
    (= sum of m over the keys), n, nn = n (n - 1) / 2, per row.

    The token-coherent terms count the pairs of distinct sources sigma != sigma' only
    (audit items b1, b2, approved 2026-10-08): kappa_NN (g^N_I g^N_I - G^N_I) for I = J
    (G^N_I = Fm2, UbN2: the pairs nu = nu' of the previous-token sources), kappa_2 (K2 - G1)
    in C_d and kappa_2 (K_gen^2 - J2) in C_all. as_written=True gives the appendix as
    written (eqs. tc_rule, Cd_bg, keyset_sums), which counts them."""
    S = Form.of
    k2, kN, kNN, pT = rates["kappa_2"], rates["kappa_N"], rates["kappa_NN"], rates["p_T"]
    Q_on, Q_T, dQ, M_off = p["Q_on"], p["Q_T"], p["dQ"], p["M_off"]
    sT2, son2, q2T, dq2 = p["var_Q_T"], p["var_Q_on"], p["q2_T"], p["dq2"]
    vMon, vMoff = p["var_M_on"], p["var_M_off"]
    D, G1, K1, K2, J2, Mall, n, nn = (bg[name] for name in ("D", "G1", "K1", "K2", "J2", "Mall", "n", "nn"))

    # eq. mean_logits: (class, mechanism, coefficient name, coefficient, statistic), h = beta/L Gamma B
    means = {
        "on": [("signal", "u", Q_on, S("Nm")), ("free", "v", Q_T, S("Fm")), ("bulk", "w", M_off * Q_T, S("W")),
               ("bulk_a", "z", M_off * dQ, S("P"))],
        "T": [("baseline_prev", "v", Q_T, Form(const=Mall)), ("baseline_bulk", "w", M_off * Q_T, Form(const=nn)),
              ("signal", "s", dQ, S("Nm")), ("bulk_a", "z", M_off * dQ, S("R"))],
        "b": [("free", "v", Q_T, S("Ub")), ("bulk", "w", M_off * Q_T, S("Wb")), ("bulk_a", "z", M_off * dQ, S("Pb"))],
    }
    brackets = {cls: sum((coef * stat for _, _, coef, stat in terms), Form()) for cls, terms in means.items()}
    B_on, B_T, B_b = brackets["on"], brackets["T"], brackets["b"]

    # eqs. split_a, AX, g_sets: the a-attentions, the generic weights by source type
    ups_P, ups_R, ups_b = S("Nm") + M_off * S("P"), S("Nm") + M_off * S("R"), M_off * S("Pb")
    K_gen = Form(const=K1) - S("Nm") - M_off * S("R")                  # K_all - Upsilon_all
    gN_phi, gB_phi = S("Fm"), M_off * (S("W") - S("P"))
    gN_b, gtr_b, gB_b = S("Ub") - S("Utr"), S("Utr"), M_off * (S("Wb") - S("Pb"))
    gB_all = Form(const=M_off * nn) - M_off * S("R")
    c_phi, c_b = S("ell") + S("f"), S("c")
    one = Form(const=1.0)

    def background(c, pairs, name_c):           # q2T c/n D + sT2 J_II  (eq. share_bg, I = J)
        return [Piece("incoherent", "", ("q2_T", "D"), q2T * D / n, c),
                Piece("source_coherent", "same_key", ("var_Q_T", "G1"), sT2 * G1 / n, c),
                Piece("source_coherent", "diff_key", ("var_Q_T", "J2-G1"), sT2 * (J2 - G1) / n ** 2, pairs)]

    def tc_self(gN, gtr, gB, same_source):     # T(I, I), eq. tc_rule (b1: without the pairs nu = nu' of NN)
        pieces = [Piece("token_coherent", "BB", ("var_Q_T", "kappa_2", "gB", "gB"), sT2 * k2, gB, gB),
                  Piece("token_coherent", "NB", ("var_Q_T", "kappa_N", "gN", "gB"), 2 * sT2 * kN, gN, gB),
                  Piece("token_coherent", "NN", ("var_Q_T", "kappa_NN", "gN", "gN"), sT2 * kNN, gN, gN,
                        linear=None if as_written else -S(same_source))]
        if gtr is not None:
            pieces.insert(2, Piece("token_coherent", "TB", ("var_Q_T", "p_T", "gtr", "gB"), 2 * sT2 * pT, gtr, gB))
        return pieces

    def tc_all(gN, gtr, gB):                   # T(I, all), eq. tc_all, without the bigram term
        pieces = [Piece("token_coherent", "B_all", ("var_Q_T", "kappa_2", "gB", "K_all"), sT2 * k2, gB, K_gen),
                  Piece("token_coherent", "N_all", ("var_Q_T", "kappa_N", "gN", "K_all"), sT2 * kN, gN, K_gen)]
        if gtr is not None:
            pieces.append(Piece("token_coherent", "tr_all", ("var_Q_T", "p_T", "gtr", "gB_all"), sT2 * pT, gtr, gB_all))
        return pieces

    def tc_cross(I, J, cross=False):           # T(I, J), I != J
        (gN_I, gtr_I, gB_I), (gN_J, gtr_J, gB_J) = I, J
        pieces = [Piece("token_coherent", "BB", ("var_Q_T", "kappa_2", "gB", "gB"), sT2 * k2, gB_I, gB_J, cross),
                  Piece("token_coherent", "NB", ("var_Q_T", "kappa_N", "gN", "gB"), sT2 * kN, gN_I, gB_J, cross),
                  Piece("token_coherent", "BN", ("var_Q_T", "kappa_N", "gB", "gN"), sT2 * kN, gB_I, gN_J, cross)]
        if gtr_I is not None:
            pieces.append(Piece("token_coherent", "TB", ("var_Q_T", "p_T", "gtr", "gB"), sT2 * pT, gtr_I, gB_J, cross))
        if gtr_J is not None:
            pieces.append(Piece("token_coherent", "BT", ("var_Q_T", "p_T", "gB", "gtr"), sT2 * pT, gB_I, gtr_J, cross))
        pieces.append(Piece("token_coherent", "NN", ("var_Q_T", "kappa_NN", "gN", "gN"), sT2 * kNN, gN_I, gN_J, cross))
        return pieces

    def a_incoherent(ell_x, X):                # eq. AX
        pieces = [] if ell_x is None else [Piece("a_incoherent", "prev", ("dq2", "var_M_on"), dq2 * vMon, ell_x)]
        return pieces + [Piece("a_incoherent", "bulk", ("dq2", "var_M_off"), dq2 * vMoff, X)]

    phi, b = (gN_phi, None, gB_phi), (gN_b, gtr_b, gB_b)
    bigram = Piece("token_coherent", "bigram", ("var_Q_T", "Y_b"), sT2, S("Yb"))
    witness = Piece("source_coherent", "witness", ("var_Q_T", "N_phi2"), -sT2, S("Nm2"))
    witness_bulk = Piece("source_coherent", "witness_bulk", ("var_Q_T", "M_off", "R_phi"), -sT2 * M_off, S("Rm"))
    keysets = {
        "C_all": [Piece("incoherent", "", ("q2_T", "D"), q2T * D, one),
                  Piece("source_coherent", "", ("var_Q_T", "J2"), sT2 * J2, one),
                  Piece("token_coherent", "BB", ("var_Q_T", "kappa_2", "K_all", "K_all"), sT2 * k2, K_gen, K_gen,
                        linear=None if as_written else Form(const=-J2)),
                  *a_incoherent(S("ell"), S("R")),
                  Piece("a_coherent", "", ("var_Q_on", "Ups_R", "Ups_R"), son2, ups_R, ups_R)],
        "C_phi": [*background(c_phi, S("pairs_phi"), "phi"), witness, *tc_self(*phi, "Fm2"),
                  *a_incoherent(S("ell"), S("P")),
                  Piece("a_coherent", "", ("var_Q_on", "Ups_P", "Ups_P"), son2, ups_P, ups_P)],
        "C_b": [*background(c_b, S("pairs_b"), "b"), *tc_self(*b, "UbN2"), bigram, *a_incoherent(None, S("Pb")),
                Piece("a_coherent", "", ("var_Q_on", "Ups_b", "Ups_b"), son2, ups_b, ups_b)],
        "C_phi_all": [Piece("incoherent", "", ("q2_T", "D"), q2T * D / n, c_phi),
                      Piece("source_coherent", "share", ("var_Q_T", "J2"), sT2 * J2 / n, c_phi),
                      witness, witness_bulk, *tc_all(gN_phi, None, gB_phi), *a_incoherent(S("ell"), S("P")),
                      Piece("a_coherent", "", ("var_Q_on", "Ups_P", "Ups_R"), son2, ups_P, ups_R)],
        "C_b_all": [Piece("incoherent", "", ("q2_T", "D"), q2T * D / n, c_b),
                    Piece("source_coherent", "share", ("var_Q_T", "J2"), sT2 * J2 / n, c_b),
                    *tc_all(*b), bigram, *a_incoherent(None, S("Pb")),
                    Piece("a_coherent", "", ("var_Q_on", "Ups_b", "Ups_R"), son2, ups_b, ups_R)],
        "C_phi_b": [Piece("source_coherent", "diff_key", ("var_Q_T", "J2-G1"), sT2 * (J2 - G1) / n ** 2, c_phi, c_b),
                    Piece("source_coherent", "witness_bulk", ("var_Q_T", "M_off", "R_phi"), -sT2 * M_off / n,
                          S("Rm"), c_b),
                    *tc_cross(phi, b),
                    Piece("a_coherent", "", ("var_Q_on", "Ups_P", "Ups_b"), son2, ups_P, ups_b)],
        "C_bb'": [Piece("source_coherent", "diff_key", ("var_Q_T", "J2-G1"), sT2 * (J2 - G1) / n ** 2, c_b, c_b, True),
                  *tc_cross(b, b, cross=True),
                  Piece("a_coherent", "", ("var_Q_on", "Ups_b", "Ups_b"), son2, ups_b, ups_b, True)],
    }
    # eq. E: the mean amplitudes, then the attention noise (C_d + kappa_2 C_o split by mechanism); the
    # token-coherent pairs of sources sigma != sigma' within a key (K2 - G1) and across keys (K1^2 - K2 - (J2 - G1))
    same_key, diff_key = (0.0, 0.0) if as_written else (G1, J2 - G1)
    energy = [Piece("mean_amplitude", "on2", ("B_on", "B_on"), 1.0, B_on, B_on),
              Piece("mean_amplitude", "same_key", ("Q_T^2", "K2"), Q_T ** 2 * K2, one),
              Piece("mean_amplitude", "X1", ("Q_T", "dQ", "M_off", "X1"), 2 * Q_T * dQ * M_off, S("X1")),
              Piece("mean_amplitude", "X2", ("dQ^2", "M_off^2", "X2"), (dQ * M_off) ** 2, S("X2")),
              Piece("mean_amplitude", "diff_key", ("kappa_2", "B_T-B_on", "B_T-B_on"), k2, B_T - B_on, B_T - B_on),
              Piece("incoherent", "same_key", ("q2_T", "D"), q2T * D, one),
              Piece("source_coherent", "same_key", ("var_Q_T", "G1"), sT2 * G1, one),
              Piece("token_coherent", "same_key", ("var_Q_T", "kappa_2", "K2-G1"), sT2 * k2 * (K2 - same_key), one),
              Piece("source_coherent", "diff_key", ("kappa_2", "var_Q_T", "J2-G1"), k2 * sT2 * (J2 - G1), one),
              Piece("token_coherent", "diff_key", ("kappa_2^2", "var_Q_T", "K1^2-K2-J2+G1"),
                    k2 ** 2 * sT2 * (K1 ** 2 - K2 - diff_key), one),
              *a_incoherent(S("ell"), S("R")),
              Piece("a_coherent", "", ("var_Q_on", "Ups_P", "Ups_P"), son2, ups_P, ups_P),
              Piece("token_coherent", "bigram", ("var_Q_T", "kappa_bi-kappa_2^2", "Mall^2"),
                    sT2 * (rates["kappa_bi"] - k2 ** 2) * Mall ** 2, one)]
    own = {"on": Piece("mean_amplitude", "own", ("B_on", "B_on"), 1.0, B_on, B_on),
           "b": Piece("mean_amplitude", "own", ("B_b", "B_b"), 1.0, B_b, B_b)}
    return {"means": means, "keysets": keysets, "energy": energy, "own": own}


# ---- assembly --------------------------------------------------------------------------------

def _readout_factors(p: dict, prefactor: float) -> dict:
    G_on, G_T = p["G_on"], p["G_T"]
    return {"G_on^2": prefactor * G_on ** 2, "G_T^2": prefactor * G_T ** 2, "G_on*G_T": prefactor * G_on * G_T,
            "var_G_on": prefactor * p["var_G_on"], "var_G_N": prefactor * p["var_G_N"],
            "var_G_T": prefactor * p["var_G_T"]}


def _expand(value, nb):
    """(rows,) -> (rows, nb), as a view."""
    return value[:, None].expand(-1, nb) if torch.is_tensor(value) and value.dim() == 1 else value


# the key sets each pair reads (pair table of the appendix)
KEYSETS_OF_PAIR = {"on,on": ("C_phi", "E", "own"), "T,T": ("C_all", "E"), "T,T'": ("C_all",), "on,T": ("C_phi_all",),
                   "b,b": ("C_b", "E", "own"), "on,b": ("C_phi_b",), "b,T": ("C_b_all",), "b,b'": ("C_bb'",)}
TRIGGER_PAIRS = ("on", "T", "b", *KEYSETS_OF_PAIR)


def _needed_keysets(pairs) -> set | None:
    if pairs is None:
        return None
    unknown = set(pairs) - set(TRIGGER_PAIRS)
    if unknown:
        raise ValueError(f"unknown pairs {sorted(unknown)}; pairs: {TRIGGER_PAIRS}")
    return {keyset for pair in pairs for keyset in KEYSETS_OF_PAIR.get(pair, ())}


def assemble_trigger(evaluated: dict, factors: dict, labels: dict, nb: int, pairs=None) -> list[Term]:
    """The noise terms of every pair (eq. pairs_trigger), or of `pairs` only. evaluated:
    {"keysets": {name: [(piece, value)]}, "energy": [(piece, value)], "own": {"on": value,
    "b": value}}, values per row; factors: the readout factors times (beta/L)^2."""
    terms = []

    def add(pair, channel, column, keyset, piece, factor_name, value, sign=1.0, label=None):
        if pairs is not None and pair not in pairs:
            return
        detail = piece.detail
        name = f"{pair}/{channel}{'.' + column if column else ''}/{keyset}.{piece.mechanism}{'.' + detail if detail else ''}"
        factor = factors[factor_name] * sign
        terms.append(Term(name=name, pair=pair, channel=channel, column=column, mechanism=piece.mechanism,
                          keyset=keyset, detail=detail, label=label or labels["keyset"],
                          params=(factor_name, *piece.params), value=factor * value))

    keysets, energy, own = evaluated["keysets"], evaluated["energy"], evaluated["own"]
    keysets = {name: keysets.get(name, []) for name in ("C_all", "C_phi", "C_b", "C_phi_all", "C_b_all", "C_phi_b", "C_bb'")}
    own = {cls: own.get(cls, (None, None)) for cls in ("on", "b")}
    # target: attention through Gamma_on, readout of its own column, readout of the other columns
    for piece, value in keysets["C_phi"]:
        add("on,on", "attention", "", "C_phi", piece, "G_on^2", value)
    add("on,on", "readout", "own", "B_on2", own["on"][0], "var_G_on", own["on"][1], label=labels["energy"])
    for piece, value in keysets["C_phi"]:
        add("on,on", "readout", "own", "C_phi", piece, "var_G_on", value)
    for piece, value in energy:
        if piece.detail != "on2":                                  # E - B_on^2 - C_phi
            add("on,on", "readout", "other", "E", piece, "var_G_N", value, label=labels["energy"])
    for piece, value in keysets["C_phi"]:
        add("on,on", "readout", "other", "C_phi", piece, "var_G_N", value, sign=-1.0)
    # triggers: common attention and individual readout of every column
    for piece, value in keysets["C_all"]:
        add("T,T", "attention", "", "C_all", piece, "G_T^2", value)
    for piece, value in energy:
        add("T,T", "readout", "all", "E", piece, "var_G_T", value, label=labels["energy"])
    for piece, value in keysets["C_all"]:
        add("T,T'", "attention", "", "C_all", piece, "G_T^2", value)
    for piece, value in keysets["C_phi_all"]:
        add("on,T", "attention", "", "C_phi_all", piece, "G_on*G_T", value)
    # non-targets
    for piece, value in keysets["C_b"]:
        add("b,b", "attention", "", "C_b", piece, "G_on^2", value)
    add("b,b", "readout", "own", "B_b2", own["b"][0], "var_G_on", own["b"][1], label=labels["energy"])
    for piece, value in keysets["C_b"]:
        add("b,b", "readout", "own", "C_b", piece, "var_G_on", value)
    for piece, value in energy:
        add("b,b", "readout", "other", "E", piece, "var_G_N", _expand(value, nb), label=labels["energy"])
    add("b,b", "readout", "other", "B_b2", own["b"][0], "var_G_N", own["b"][1], sign=-1.0, label=labels["energy"])
    for piece, value in keysets["C_b"]:
        add("b,b", "readout", "other", "C_b", piece, "var_G_N", value, sign=-1.0)
    for piece, value in keysets["C_phi_b"]:
        add("on,b", "attention", "", "C_phi_b", piece, "G_on^2", value)
    for piece, value in keysets["C_b_all"]:
        add("b,T", "attention", "", "C_b_all", piece, "G_on*G_T", value)
    for piece, value in keysets["C_bb'"]:
        add("b,b'", "attention", "", "C_bb'", piece, "G_on^2", value)
    return terms


def _mean_terms(structure, evaluator, p, beta, L, label, pairs=None) -> list[Term]:
    terms = []
    for cls, entries in structure["means"].items():
        if pairs is not None and cls not in pairs:
            continue
        gamma = p["G_T"] if cls == "T" else p["G_on"]
        for mechanism, coefficient, coef, stat in entries:
            value = (beta / L) * gamma * evaluator.piece(Piece(mechanism, "", (), coef, stat))
            terms.append(Term(name=f"{cls}/mean/{mechanism}", pair=cls, channel="mean", mechanism=mechanism,
                              label=label, params=("G_T" if cls == "T" else "G_on", coefficient), value=value))
    return terms


def _spread_terms(structure, evaluator: GivenEll, p, beta, L, pairs=None) -> list[Term]:
    """Cov(hbar_X, hbar_Y | ell) term by term (eqs. var_logits, cov_logits): one term per pair
    of mean terms whose statistics covary; for a variance the pairs i < j count twice."""
    terms = []
    means = structure["means"]
    k = beta / L
    gamma = {"on": p["G_on"], "T": p["G_T"], "b": p["G_on"]}
    for X, Y, pair in (("on", "on", "on,on"), ("T", "T", "T,T"), ("T", "T", "T,T'"), ("on", "T", "on,T"),
                       ("b", "b", "b,b"), ("on", "b", "on,b"), ("b", "T", "b,T"), ("b", "b", "b,b'")):
        if pairs is not None and pair not in pairs:
            continue
        same = X == Y
        label = ELL_LABELS["var"] if same and pair != "b,b'" else ELL_LABELS["cov"]
        for i, (mech_i, coef_name_i, coef_i, stat_i) in enumerate(means[X]):
            for j, (mech_j, coef_name_j, coef_j, stat_j) in enumerate(means[Y]):
                if same and j < i:
                    continue
                factor = (k ** 2) * gamma[X] * gamma[Y] * coef_i * coef_j * (2.0 if same and j > i else 1.0)
                if pair == "b,b'":
                    piece = Piece(f"{mech_i}*{mech_j}", "", (), factor, stat_i, stat_j, cross=True)
                    value = evaluator.cross(piece)
                    value.parts = value.parts[1:]                 # keep the covariance, drop E X E Y
                    if not value.parts:
                        continue
                else:
                    covariance = evaluator.cov(stat_i, stat_j)
                    if isinstance(covariance, float) and covariance == 0.0:
                        continue
                    kind = "b" if "b" in (stat_i.kind, stat_j.kind) else "t"
                    value = (factor[:, None] if kind == "b" and torch.is_tensor(factor) and factor.dim() == 1
                             else factor) * covariance
                terms.append(Term(name=f"{pair}/spread/{mech_i}*{mech_j}", pair=pair, channel="spread",
                                  mechanism=f"{mech_i}*{mech_j}", label=label,
                                  params=(coef_name_i, coef_name_j), value=value))
    return terms


def _evaluate(structure, evaluator, needed=None) -> dict:
    """Every piece evaluated per row; only the key sets in `needed` (None: all)."""
    use = lambda name: needed is None or name in needed
    return {"keysets": {name: [(piece, evaluator.piece(piece)) for piece in pieces]
                        for name, pieces in structure["keysets"].items() if use(name)},
            "energy": [(piece, evaluator.piece(piece)) for piece in structure["energy"]] if use("E") else [],
            "own": {cls: (piece, evaluator.piece(piece)) for cls, piece in structure["own"].items() if use("own")}}


def _tensors(order_params: dict, dtype, device) -> dict:
    return _derived(_parameters(order_params, dtype, device))


def _background(diag, p, n_keys, dtype) -> dict:
    sums = _key_sums(diag, p["M_off"], n_keys, p["var_M_on"], p["var_M_off"])
    n_float = n_keys.to(dtype)
    return {"D": sums["D"], "G1": sums["G1"], "K1": sums["K1"], "K2": sums["K2"], "J2": sums["J2"],
            "Mall": sums["all"], "n": n_float.clamp(min=1), "nn": n_float * (n_float - 1) / 2}


# ---- given S ------------------------------------------------------------------------------

def trigger_statistics(variables: QueryTable, order_params: dict, L: int) -> dict:
    """The statistics of `forms` at every row of a table of variables (measured, or sampled;
    the key columns are needed). Differentiable in the profile (or M_on) and M_off.

    The predecessor type of the keys ("U_triggered": measured, or sampled with
    triggered=True) enters only Utr, Yb, UbN2. Without it they are replaced by their mean
    over the types given the keys (each key of an output follows its trigger with
    probability 1/2): Utr = S1/2, Yb = (S1^2 - S2)/4 with S1, S2 the sums of m, m^2 over the
    output's keys, and UbN2 = S2/4, the value that also makes the NN term,
    kappa_NN (g^N^2 - UbN2), equal to its conditional mean. Every noise term is then the
    exact mean over the types given the keys."""
    dtype, device = variables["N"].dtype, variables["N"].device
    M_off = torch.as_tensor(order_params.get("M_off", 0.0), dtype=dtype, device=device)
    diag = _diagonal(order_params, L, dtype, device)
    n_keys = (variables["mu"] - 2).clamp(min=0)
    witness_keys = variables["N_keys"]
    m_witness = _take(diag, witness_keys)
    witness = _witness_sums(witness_keys, diag, M_off, n_keys)
    m_tokens = _take(diag, variables["U_keys"])
    ell, f, c = variables["N"], variables["F"], variables["U_bar"]
    m_free = _take(diag, variables["F_keys"])
    Utr, Yb, UbN2 = _predecessor_sums(variables, m_tokens, int(variables["K"]) - 1)
    return {"ell": ell, "f": f, "Nm": m_witness.sum(1), "Fm": m_free.sum(1), "Fm2": (m_free ** 2).sum(1),
            "W": variables["W"], "P": variables["P"], "R": variables["R"], "Nm2": (m_witness ** 2).sum(1),
            "Rm": (m_witness * (n_keys[:, None] - witness_keys) * (witness_keys > 0)).sum(1),
            "X1": witness["X1"], "X2": witness["X2"], "pairs_phi": (ell + f) * (ell + f - 1),
            "c": c, "Ub": m_tokens.sum(-1), "Utr": Utr, "Wb": variables["W_bar"], "Pb": variables["P_bar"],
            "Yb": Yb, "pairs_b": c * (c - 1), "UbN2": UbN2}


def _predecessor_sums(variables, m_tokens, num_outputs: int):
    """(Utr, Yb, UbN2) per row and token from the keys' predecessor types, or their mean over
    the types given the keys if "U_triggered" is absent (see `trigger_statistics`)."""
    squares = m_tokens ** 2
    if "U_triggered" in variables:
        triggered = variables["U_triggered"].to(m_tokens.dtype)
        Utr = (m_tokens * triggered).sum(-1)
        return Utr, Utr ** 2 - (squares * triggered).sum(-1), (squares * (1 - triggered)).sum(-1)
    is_output = (torch.arange(m_tokens.size(1), device=m_tokens.device) < num_outputs).to(m_tokens.dtype)
    S1, S2 = m_tokens.sum(-1), squares.sum(-1)
    return is_output * S1 / 2, is_output * (S1 ** 2 - S2) / 4, S2 * (1 - 3 * is_output / 4)


def trigger_terms(variables: QueryTable, order_params: dict, beta: float, L: int, pairs=None,
                  as_written: bool = False) -> TermTable:
    """Every term of the logit moments at the trigger queries of `variables`, given S_mu:
    the means (eq. mean_logits, exact) and the covariance of every pair of classes (eq.
    pairs_trigger with eqs. keyset_sums, E and the corrected rule tc_rule). Rows are the
    rows of `variables`. A missing order parameter or variance counts as 0.

    pairs: compute only these classes / pairs (e.g. ("on,on", "b,b")); None = all.
    as_written: the appendix as written, which counts the same-source pairs in the
    token-coherent terms (see `_structure`); the default leaves them out."""
    dtype, device = variables["N"].dtype, variables["N"].device
    K = int(variables["K"])
    nb = variables["U_bar"].size(1)
    V = K + 1 + nb
    p = _tensors(order_params, dtype, device)
    diag = _diagonal(order_params, L, dtype, device)
    n_keys = (variables["mu"] - 2).clamp(min=0)
    structure = _structure(p, _background(diag, p, n_keys, dtype), coincidence_rates(V, K), as_written)
    evaluator = GivenS(trigger_statistics(variables, order_params, L), nb)
    terms = _mean_terms(structure, evaluator, p, beta, L, S_LABELS["mean"], pairs)
    terms += assemble_trigger(_evaluate(structure, evaluator, _needed_keysets(pairs)),
                              _readout_factors(p, (beta / L) ** 2), S_LABELS, nb, pairs)
    rows = QueryTable({"mu": variables["mu"], "ell": variables["ell"], "K": K})
    return TermTable(terms, rows, "S", "trigger", K, V)


# ---- given ell ----------------------------------------------------------------------------

def window_moments(order_params: dict, mu, L: int, dtype=torch.float64, device=None) -> dict:
    """The window moments of the m-weighted profile over the keys of a query at mu (eq.
    window_moments, discrete): Phi1 = mean of m_nu, Phi2 = mean of m_nu^2, Psi = mean of
    m_nu x_nu with x_nu = (j - 1/2) / n the relative position of the key at code position j
    (the convention of `sample_variables`). Without a profile m = M_on, so Phi1 = M_on,
    Phi2 = M_on^2, Psi = M_on / 2."""
    mu = torch.as_tensor(mu, device=device).reshape(-1)
    diag = _diagonal(order_params, L, dtype, device)
    n_keys = (mu - 2).clamp(min=0)
    n = n_keys.to(dtype).clamp(min=1)
    position = torch.arange(len(diag), dtype=dtype, device=diag.device)
    s1, s2, sj = (_take(values.cumsum(0), n_keys) for values in (diag, diag ** 2, position * diag))
    return {"Phi1": s1 / n, "Phi2": s2 / n, "Psi": (sj - s1 / 2) / n ** 2}


def _leading_background(window, p, mu, n_keys, dtype, exact) -> dict:
    """eq. sums_bg in the m-weighted window moments (Mall, n, nn stay exact)."""
    P1, P2, Psi = window["Phi1"], window["Phi2"], window["Psi"]
    M_off = p["M_off"]
    background = dict(exact)
    background.update(D=p["var_M_on"] * mu + p["var_M_off"] * mu ** 2 / 2, K1=mu * (P1 + M_off * mu / 2),
                      J2=mu * (P2 + 2 * M_off * mu * (P1 - Psi) + M_off ** 2 * mu ** 2 / 3),
                      G1=mu * (P2 + M_off ** 2 * mu / 2), K2=mu * (P2 + 2 * M_off * mu * Psi + M_off ** 2 * mu ** 2 / 3))
    return background


def _ell_setup(mu, ell, order_params, L, V, K, nb, outputs, Lambda, background, dtype, device):
    if Lambda not in LAMBDAS:
        raise ValueError(f"Lambda must be one of {LAMBDAS}, got {Lambda!r}")
    if background not in BACKGROUNDS:
        raise ValueError(f"background must be one of {BACKGROUNDS}, got {background!r}")
    mu, ell = torch.broadcast_tensors(torch.as_tensor(mu, device=device).reshape(-1).long(),
                                      torch.as_tensor(ell, device=device).reshape(-1).long())
    mu_f, ell_f = mu.to(dtype), ell.to(dtype)
    p = _tensors(order_params, dtype, device)
    p_T = 1 / (V + K)
    n_keys = (mu - 2).clamp(min=0)
    lam = p_T * mu_f if Lambda == "mu" else p_T * (n_keys.to(dtype) - 2 * ell_f).clamp(min=0)
    s = (torch.arange(nb, device=device) < outputs).to(dtype)
    r = lam[:, None] * (1 + s)
    window = window_moments(order_params, mu, L, dtype, device)
    diag = _diagonal(order_params, L, dtype, device)
    bg = _background(diag, p, n_keys, dtype)
    if background == "leading":
        bg = _leading_background(window, p, mu_f, n_keys, dtype, bg)
    X1_mean = mu_f * ell_f * (window["Psi"] + p["M_off"] * mu_f / 3)
    evaluator = GivenEll(mu_f, ell_f, lam, r, s, window, p["M_off"], X1_mean)
    return mu, ell, p, bg, evaluator


def trigger_terms_ell(mu, ell, order_params: dict, beta: float, L: int, V: int, K: int, Lambda: str = "mu",
                      background: str = "exact", dtype=torch.float64, device=None, pairs=None,
                      as_written: bool = False, classes_only: bool = False) -> TermTable:
    """Every term of the logit moments at a trigger query given (mu, ell) (ints or 1-D
    tensors, broadcast; one row per pair): the means (eq. mean_logits_ell), the spread of
    the means (eqs. var_logits, cov_logits, channel "spread") and the mean noise (eq.
    pairs_trigger_ell with eqs. keyset_means, E_mean, tc_mean), so that the covariance given
    ell (eq. cov_total) is the total over both channels.

    Each noise term is the conditional mean of the term of the same name of
    `trigger_terms` under A1-A3. Lambda: "mu" (p_T mu, as `mean_variables`) or "n-2ell"
    (p_T (mu - 2 - 2 ell), the finite-size correction of the appendix). background:
    "exact" (the discrete sums D, G1, K1, K2, J2 of the mean M) or "leading" (eq. sums_bg).
    The window moments are those of `window_moments`. pairs, as_written: as `trigger_terms`.
    classes_only: the b-indexed values for one output and one plain token only, columns
    (0, 1) (they are equal within a class); then "b,b'" is not available."""
    if classes_only:
        if pairs is not None and "b,b'" in pairs:
            raise ValueError("classes_only has no b,b' covariance")
        pairs = [pair for pair in TRIGGER_PAIRS if pair != "b,b'"] if pairs is None else pairs
    nb, outputs = (2, 1) if classes_only else (V - K - 1, K - 1)
    mu, ell, p, bg, evaluator = _ell_setup(mu, ell, order_params, L, V, K, nb, outputs, Lambda, background, dtype, device)
    structure = _structure(p, bg, coincidence_rates(V, K), as_written)
    terms = _mean_terms(structure, evaluator, p, beta, L, ELL_LABELS["mean"], pairs)
    terms += _spread_terms(structure, evaluator, p, beta, L, pairs)
    terms += assemble_trigger(_evaluate(structure, evaluator, _needed_keysets(pairs)),
                              _readout_factors(p, (beta / L) ** 2), ELL_LABELS, nb, pairs)
    classes = torch.eye(2, dtype=torch.bool) if classes_only else None
    return TermTable(terms, QueryTable({"mu": mu, "ell": ell, "K": K}), "ell", "trigger", K, V, b_classes=classes)
