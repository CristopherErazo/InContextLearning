"""The exact covariance of the logits given the sequence (eqs. cov_exact, CIJ, Calpha,
column_energy), split by mechanism, at the trigger (`exact_trigger_terms`) or non-trigger
(`exact_nontrigger_terms`) queries of a batch. Mainly the arbiter of the tests and of the
audit: its terms carry the same tags as those of `trigger_terms` / `nontrigger_terms`, so
the two levels compare mechanism by mechanism through `TermTable.aggregate`.

The key-set sums C(I_c, I_c') = sum over keys of C^alpha are split by where the noise comes
from (eq. split_a): for a query row a of Q,

    incoherent       q2_T sum_{nu in I cap J} sum_sigma var_M      (every source as generic)
    a_incoherent     dq2 sum_{nu in I cap J} sum_{sigma: tau_sigma = a} var_M
    a_coherent       var_Q_on Upsilon_I Upsilon_J
    source_coherent  var_Q_T sum_{sigma: tau_sigma != a} T_I(sigma) T_J(sigma)
    token_coherent   var_Q_T sum_{sigma != sigma', tau_sigma = tau_sigma' != a} T_I(sigma) T_J(sigma')

(at a non-trigger query no source is special and var_Q_N replaces both), and the column
energy into its mean amplitudes sum_c B_c^2 and these noise parts. The b-b' covariances are
kept as class sums (`ClassPairSums`). Means are not included: those of `trigger_terms` are
already exact.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..sums import _diagonal
from ..query_table import QueryTable, canonical_permutation, trigger_queries
from ..variables import nontrigger_permutation, nontrigger_queries
from .forms import Piece
from .nontrigger import assemble_nontrigger
from .table import ClassPairSums, TermTable
from .trigger import _readout_factors, _tensors, assemble_trigger

MECHANISMS = ("incoherent", "a_incoherent", "a_coherent", "source_coherent", "token_coherent")
LABEL = "eq-apx:cov_exact"


def _keyset_matrices(tokens, query, mu: int, diag, p, trigger: bool, V: int) -> tuple[dict, torch.Tensor]:
    """{mechanism: C(I_c, I_c') (rows, V, V)} in token order and the brackets B_c (rows, V)
    for queries at mu (tokens: (rows, >= mu - 1) code positions)."""
    dtype = diag.dtype
    n = mu - 2
    sources, keys = tokens[:, :n], tokens[:, 1:n + 1]
    lower = torch.ones(n, n, dtype=dtype, device=diag.device).tril(-1)
    eye = torch.eye(n, dtype=dtype, device=diag.device)
    M = torch.diag(diag[1:n + 1]) + p["M_off"] * lower                   # [key, source]
    var_M = p["var_M_on"] * eye + p["var_M_off"] * lower
    holds = F.one_hot(keys, V).to(dtype)                                 # (rows, keys, V)
    source_holds = F.one_hot(sources, V).to(dtype)
    T = torch.einsum("js,rjc->rsc", M, holds)                            # T_{I_c}(sigma)
    U = torch.einsum("rsc,rsd->rcd", source_holds, T)                    # sum of T over the sources holding c
    per_key = var_M.sum(1)
    if not trigger:
        variance = p["var_Q_N"]
        same_source = variance * torch.einsum("rsc,rsd->rcd", T, T)
        coherent = variance * torch.einsum("rcd,rce->rde", U, U)
        incoherent = torch.diag_embed(torch.einsum("rjc,j->rc", holds, variance * per_key))
        zero = torch.zeros_like(coherent)
        return ({"incoherent": incoherent, "a_incoherent": zero, "a_coherent": zero,
                 "source_coherent": same_source, "token_coherent": coherent - same_source},
                torch.zeros(len(tokens), V, dtype=dtype, device=diag.device))
    is_a = (sources == query[:, None]).to(dtype)
    q_bar = p["Q_T"] + p["dQ"] * is_a
    amplitude = q_bar @ M.T                                             # A_bar_nu
    incoherent = torch.diag_embed(torch.einsum("rjc,j->rc", holds, p["q2_T"] * per_key))
    a_incoherent = torch.diag_embed(torch.einsum("rjc,rj->rc", holds, p["dq2"] * (is_a @ var_M.T)))
    same_source = p["var_Q_T"] * torch.einsum("rsc,rs,rsd->rcd", T, 1 - is_a, T)
    not_a = 1 - F.one_hot(query, V).to(dtype)
    coherent = p["var_Q_T"] * torch.einsum("rcd,rc,rce->rde", U, not_a, U)
    upsilon = U[torch.arange(len(query), device=query.device), query]   # (rows, V)
    a_coherent = p["var_Q_on"] * upsilon[:, :, None] * upsilon[:, None, :]
    brackets = torch.einsum("rjc,rj->rc", holds, amplitude)
    return ({"incoherent": incoherent, "a_incoherent": a_incoherent, "a_coherent": a_coherent,
             "source_coherent": same_source, "token_coherent": coherent - same_source}, brackets)


def _class_sums(block: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    """(rows, k, k) sums over b in class i, b' in class j, b != b' of a (rows, nb, nb) block."""
    masks = classes.to(block.dtype)
    off_diagonal = block - torch.diag_embed(torch.diagonal(block, dim1=1, dim2=2))
    return torch.einsum("ib,rbc,jc->rij", masks, off_diagonal, masks)


def _reduce_trigger(matrices, brackets, permutation, K, V, classes) -> dict:
    rows = torch.arange(len(permutation), device=permutation.device)
    index = (rows[:, None, None], permutation[:, :, None], permutation[:, None, :])
    nt = slice(K, V - 1)
    out = {}
    for mechanism, matrix in matrices.items():
        C = matrix[index]
        diagonal = torch.diagonal(C, dim1=1, dim2=2)
        out[("C_phi", mechanism)] = C[:, -1, -1]
        out[("C_all", mechanism)] = C.sum((1, 2))
        out[("C_phi_all", mechanism)] = C[:, -1, :].sum(-1)
        out[("C_b", mechanism)] = diagonal[:, nt]
        out[("C_b_all", mechanism)] = C[:, nt, :].sum(-1)
        out[("C_phi_b", mechanism)] = C[:, -1, nt]
        out[("C_bb'", mechanism)] = _class_sums(C[:, nt, nt], classes)
        out[("E", mechanism)] = diagonal.sum(1)
    B = brackets.gather(1, permutation)
    out[("E", "on2")] = B[:, -1] ** 2
    out[("E", "rest")] = (B ** 2).sum(1) - B[:, -1] ** 2
    out[("own", "on")] = B[:, -1] ** 2
    out[("own", "b")] = B[:, nt] ** 2
    return out


def _reduce_nontrigger(matrices, permutation, K, V, classes) -> dict:
    rows = torch.arange(len(permutation), device=permutation.device)
    index = (rows[:, None, None], permutation[:, :, None], permutation[:, None, :])
    rho = slice(K, V)
    out = {}
    for mechanism in ("incoherent", "source_coherent", "token_coherent"):
        C = matrices[mechanism][index]
        diagonal = torch.diagonal(C, dim1=1, dim2=2)
        out[("C_all", mechanism)] = C.sum((1, 2))
        out[("C_rho", mechanism)] = diagonal[:, rho]
        out[("C_rho_all", mechanism)] = C[:, rho, :].sum(-1)
        out[("C_rhorho'", mechanism)] = _class_sums(C[:, rho, rho], classes)
        out[("E", mechanism)] = diagonal.sum(1)
    return out


def _run(batch, mus, order_params, L, chunk, trigger: bool):
    K, V = int(batch["K"]), int(batch["V"])
    inputs = batch["sequence"][:, :-1]
    device = inputs.device
    sequence_index, position = (trigger_queries if trigger else nontrigger_queries)(inputs, K, mus)
    if len(sequence_index) == 0:
        raise ValueError(f"no {'trigger' if trigger else 'non-trigger'} query at mu = {mus}")
    dtype = torch.float64
    p = _tensors(order_params, dtype, device)
    diag = _diagonal(order_params, L, dtype, device)
    nb = V - K - 1 if trigger else V - K
    outputs = torch.arange(nb, device=device) < (K - 1 if trigger else K)
    classes = torch.stack([outputs, ~outputs])
    pieces, order = [], []
    for mu in position.unique().tolist():
        selected = torch.where(position == mu)[0]
        mu = mu + 1
        for start in range(0, len(selected), chunk):
            rows = selected[start:start + chunk]
            tokens = inputs[sequence_index[rows]]
            query = tokens[:, mu - 1]
            matrices, brackets = _keyset_matrices(tokens, query, mu, diag, p, trigger, V)
            if trigger:
                permutation = canonical_permutation(query, batch["output_set"][sequence_index[rows]], K, V)
                pieces.append(_reduce_trigger(matrices, brackets, permutation, K, V, classes))
            else:
                permutation = nontrigger_permutation(batch["output_set"][sequence_index[rows]], K, V)
                pieces.append(_reduce_nontrigger(matrices, permutation, K, V, classes))
            order.append(rows)
    inverse = torch.empty(len(sequence_index), dtype=torch.long, device=device)
    inverse[torch.cat(order)] = torch.arange(len(sequence_index), device=device)
    values = {key: torch.cat([piece[key] for piece in pieces])[inverse] for key in pieces[0]}
    query_tokens = inputs[sequence_index, position]
    same = inputs[sequence_index] == query_tokens[:, None]                 # earlier occurrences of the query token
    ell = same.cumsum(1).gather(1, position[:, None]).squeeze(1) - 1
    rows = QueryTable({"mu": position + 1, "ell": ell, "sequence_index": sequence_index, "K": K})
    return values, rows, p, K, V, nb


def _evaluated(values: dict) -> dict:
    evaluated = {"keysets": {}, "energy": [], "own": {}}
    for (keyset, mechanism), value in values.items():
        if keyset == "own":
            continue
        if keyset.startswith("C_") and keyset.endswith("'"):
            value = ClassPairSums(value)
        if keyset == "E":
            detail = mechanism if mechanism in ("on2", "rest") else ""
            evaluated["energy"].append((Piece("mean_amplitude" if detail else mechanism, detail, (), None, None), value))
        else:
            evaluated["keysets"].setdefault(keyset, []).append((Piece(mechanism, "", (), None, None), value))
    return evaluated


def exact_trigger_terms(batch: dict, mus, order_params: dict, beta: float, L: int, chunk: int = 64) -> TermTable:
    """The exact covariance given the sequence at every trigger query of `batch` at the
    chosen mu, as terms (pairs and tags of `trigger_terms`; mechanisms only, no details).
    Rows line up with `measure_variables(batch, mus)`. Differentiable in the order
    parameters; cost ~ rows x mu x V^2."""
    values, rows, p, K, V, nb = _run(batch, mus, order_params, L, chunk, trigger=True)
    evaluated = _evaluated(values)
    evaluated["own"] = {"on": (Piece("mean_amplitude", "own", (), None, None), values[("own", "on")]),
                        "b": (Piece("mean_amplitude", "own", (), None, None), values[("own", "b")])}
    labels = {"keyset": LABEL, "energy": LABEL}
    terms = assemble_trigger(evaluated, _readout_factors(p, (beta / L) ** 2), labels, nb)
    return TermTable(terms, rows, "tau", "trigger", K, V)


def exact_nontrigger_terms(batch: dict, mus, order_params: dict, beta: float, L: int, chunk: int = 64) -> TermTable:
    """The exact covariance given the sequence at every non-trigger query of `batch` at the
    chosen mu, as terms (pairs and tags of `nontrigger_terms`). Rows line up with
    `measure_nontrigger_variables(batch, mus)`."""
    values, rows, p, K, V, nb = _run(batch, mus, order_params, L, chunk, trigger=False)
    terms = assemble_nontrigger(_evaluated(values), _readout_factors(p, (beta / L) ** 2), LABEL, nb)
    return TermTable(terms, rows, "tau", "nontrigger", K, V)
