"""The sequence variables the ansatz logits depend on (Table 1 of
paper/scratch/extended_ansatz.tex), at a trigger query tau_mu = a, with the
paper's names:

    N      earlier occurrences of a                          sum_{nu<mu} d(tau_nu, a)
    F      free occurrences of phi(a) (not right after a), over the keys
    R      sum over the earlier a's of the keys at distance >= 2 after it
    W      sum over the keys holding phi(a) of (nu - 2)
    P      pairs (a, ..., phi(a)) at distance >= 2
    U_bar, W_bar, P_bar
           the same three counts (occurrences, sum of nu - 2, pairs with an
           earlier a) for every non-trigger token other than phi(a), as
           (num_rows, V-K-1) columns in the order of the `non_target` block

Every function returns them as a `QueryTable` with "mu", "ell", these columns
and the metadata "K", ready for `ansatz_logits`.
"""
from __future__ import annotations

import torch

from .query_table import QueryTable, canonical_permutation, logit_blocks, trigger_queries


def measure_variables(batch: dict, mus=None, chunk: int = 8192, dtype=torch.float64) -> QueryTable:
    """The variables measured exactly (finite mu, no approximation) at every
    trigger query of `batch` at the chosen mu (int, list, or None = every mu).
    Its rows line up one to one with `logit_table(model, batch, mus)`.

    Only `sequence`, `output_set`, `K` and `V` of the batch are read. Works on
    the batch's device, `chunk` rows at a time; the result is on that device.
    """
    K, V = int(batch["K"]), int(batch["V"])
    inputs = batch["sequence"][:, :-1]
    L = inputs.size(1)
    all_sequence_index, all_position = trigger_queries(inputs, K, mus)
    if len(all_sequence_index) == 0:
        raise ValueError(f"no trigger query at mu = {mus}")
    code_position = torch.arange(L, device=inputs.device)[None, :]      # j, i.e. nu = j + 1
    blocks = logit_blocks(K, V)
    parts = []
    for start in range(0, len(all_sequence_index), chunk):
        sequence_index = all_sequence_index[start:start + chunk]
        position = all_position[start:start + chunk][:, None]            # the query, mu = position + 1
        tokens, output_sets = inputs[sequence_index], batch["output_set"][sequence_index]
        query_token = tokens.gather(1, position).squeeze(1)
        before_query = code_position < position                          # nu < mu
        is_key = (code_position >= 1) & before_query                     # keys: nu = 2..mu-1
        is_earlier_query = (tokens == query_token[:, None]) & before_query
        N = is_earlier_query.sum(1).to(dtype)
        R = (is_earlier_query * (position - code_position - 2).clamp(min=0)).sum(1).to(dtype)
        # n_a(nu - 2) at each key: earlier a's at code positions <= j - 2
        earlier_queries_behind = torch.zeros_like(tokens, dtype=dtype)
        earlier_queries_behind[:, 2:] = is_earlier_query.to(dtype).cumsum(1)[:, :-2]
        weight_per_key = {
            "U": is_key.to(dtype),                                       # occurrences
            "W": is_key * (code_position - 1).to(dtype),                 # nu - 2
            "P": is_key * earlier_queries_behind,                        # pairs with an earlier a
        }
        permutation = canonical_permutation(query_token, output_sets, K, V)
        per_token = {}
        for name, weight in weight_per_key.items():                     # sum of the weight over each token's keys
            totals = torch.zeros(len(sequence_index), V, dtype=dtype, device=inputs.device)
            per_token[name] = totals.scatter_add_(1, tokens, weight).gather(1, permutation)
        target, non_target = blocks["target"], blocks["non_target"]
        parts.append(QueryTable({
            "mu": position.squeeze(1) + 1, "ell": N.long(), "N": N,
            "F": per_token["U"][:, target].squeeze(1) - N,               # every earlier a is followed by phi(a)
            "R": R,
            "W": per_token["W"][:, target].squeeze(1),
            "P": per_token["P"][:, target].squeeze(1),
            "U_bar": per_token["U"][:, non_target],
            "W_bar": per_token["W"][:, non_target],
            "P_bar": per_token["P"][:, non_target],
            "K": K,
        }))
    return QueryTable.concatenate(parts)
