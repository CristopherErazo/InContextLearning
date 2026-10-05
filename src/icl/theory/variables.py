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
    Y_bar  the ordered pairs of occurrences of each non-target token that both
           follow its trigger, c_tr (c_tr - 1), (num_rows, V-K-1): 0 for the
           tokens that are no trigger's output (Y_b of the variance note, for the
           logit noise only)

`measure_variables` and `sample_variables` also return where the occurrences
are, for a previous-token *profile* ("M_profile" in the order parameters):
"N_keys" (n, max ell) the witness keys, "F_keys" (n, max f) the free target
keys and "U_keys" (n, V-K-1, max count) the keys of each non-target token, as
code positions of keys (1..mu-2), padded with 0.

Three sources, all returning a `QueryTable` with "mu", "ell", these columns and
the metadata "K", ready for `ansatz_logits`:

- `measure_variables(batch)`: measured exactly on real sequences;
- `sample_variables(mu, ell, ...)`: drawn from the law of the scratch file at
  fixed (mu, ell) (section "Sampling from the logit distribution");
- `mean_variables(mu, ell, ...)`: their means under that law (Table 2).

The law uses the paper's approximations (mu >> 1, ell << mu, positions as
independent uniforms in (0, 1)), with p_T = 1/(V + K) the stationary weight of a
trigger and m = p_T mu. A non-target token b occurs at rate r_b = 2m if it is
the output of another trigger (the `other_outputs` block) and r_b = m otherwise
(the `rest` block).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data import occurrence_counts
from .query_table import QueryTable, canonical_permutation, logit_blocks, trigger_queries

KEY_COLUMNS = ("N_keys", "F_keys", "U_keys")


def _packed(keys: torch.Tensor) -> torch.Tensor:
    """(n, L) code positions, 0 where absent -> (n, max present), the positions
    first and 0 as padding."""
    width = max(int((keys > 0).sum(1).max().item()), 1)
    return keys.sort(dim=1, descending=True).values[:, :width]


def _concatenate_padded(parts: list[QueryTable]) -> QueryTable:
    """`QueryTable.concatenate`, after padding the key columns of every part to
    the same width with 0."""
    for name in KEY_COLUMNS:
        if name in parts[0]:
            width = max(part[name].size(-1) for part in parts)
            for part in parts:
                part[name] = F.pad(part[name], (0, width - part[name].size(-1)))
    return QueryTable.concatenate(parts)


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
        follows_trigger = torch.zeros_like(is_key)
        follows_trigger[:, 1:] = tokens[:, :-1] < K
        weight_per_key = {
            "U": is_key.to(dtype),                                       # occurrences
            "W": is_key * (code_position - 1).to(dtype),                 # nu - 2
            "P": is_key * earlier_queries_behind,                        # pairs with an earlier a
            "T": (is_key & follows_trigger).to(dtype),                   # occurrences right after a trigger
        }
        permutation = canonical_permutation(query_token, output_sets, K, V)
        # where the occurrences are, for a profile: witness keys (one after each earlier a),
        # free target keys, and the keys of each non-target token in canonical order
        target_token = output_sets.gather(1, query_token[:, None])
        after_query = torch.zeros_like(is_key)
        after_query[:, 1:] = tokens[:, :-1] == query_token[:, None]
        is_free_target = is_key & (tokens == target_token) & ~after_query
        slot = torch.empty_like(permutation).scatter_(1, permutation, torch.arange(V, device=inputs.device).expand_as(permutation))
        slot_of_key = slot.gather(1, tokens)
        is_token_key = is_key & (slot_of_key >= K) & (slot_of_key < V - 1)
        rank = occurrence_counts(torch.where(is_token_key, tokens, V)) - 1      # occurrence number among its keys
        rows, columns = torch.where(is_token_key)
        width = int(rank[rows, columns].max().item()) + 1 if len(rows) else 1
        U_keys = torch.zeros(len(sequence_index), V - K - 1, width, dtype=torch.long, device=inputs.device)
        U_keys[rows, slot_of_key[rows, columns] - K, rank[rows, columns]] = columns
        keys = {"N_keys": _packed(torch.where(is_earlier_query, code_position + 1, 0)),
                "F_keys": _packed(torch.where(is_free_target, code_position, 0)),
                "U_keys": U_keys}
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
            "Y_bar": per_token["T"][:, non_target] * (per_token["T"][:, non_target] - 1),
            **keys,
            "K": K,
        }))
    return _concatenate_padded(parts)


COUNT_LAWS = ("poisson", "multinomial")


def _as_rows(mu, ell, num_samples, device) -> tuple[torch.Tensor, torch.Tensor]:
    """mu and ell (ints or 1-D tensors) broadcast to `num_samples` rows (default:
    their common length, 1 for two ints)."""
    mu, ell = (torch.as_tensor(value, device=device).long().reshape(-1) for value in (mu, ell))
    num_rows = num_samples or max(len(mu), len(ell))
    mu, ell = mu.expand(num_rows) if len(mu) == 1 else mu, ell.expand(num_rows) if len(ell) == 1 else ell
    if len(mu) != num_rows or len(ell) != num_rows:
        raise ValueError(f"mu ({len(mu)}), ell ({len(ell)}) and num_samples ({num_samples}) do not broadcast")
    if (mu < 1).any() or (ell < 0).any():
        raise ValueError("need mu >= 1 and ell >= 0")
    return mu, ell


def _non_target_rates(m: torch.Tensor, K: int, V: int) -> torch.Tensor:
    """(num_rows, V-K-1) occurrence rates r_b in the `non_target` block order."""
    return torch.cat([2 * m[:, None].expand(-1, K - 1), m[:, None].expand(-1, V - 2 * K)], dim=1)


def _uniform_positions(counts: torch.Tensor, generator, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """For integer counts of any shape: (positions, is_real) of shape
    counts.shape + (max count,), positions i.i.d. U(0, 1) where is_real."""
    slots = max(int(counts.max().item()), 1)
    positions = torch.rand(*counts.shape, slots, generator=generator, dtype=dtype, device=counts.device)
    is_real = torch.arange(slots, device=counts.device) < counts[..., None]
    return positions, is_real


def _earlier_queries_behind(sorted_query_positions: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """For positions (num_rows, ...) in (0, 1): how many earlier queries x_k lie
    before each of them. `sorted_query_positions` is (num_rows, slots), sorted,
    padded with +inf."""
    flat = positions.reshape(positions.size(0), -1).contiguous()
    behind = torch.searchsorted(sorted_query_positions, flat)
    return behind.reshape(positions.shape).to(positions.dtype)


def _multinomial_counts(mu, ell, K, V, generator, dtype):
    """Joint counts on the mu - 2 - 2 ell available keys: each key holds one
    token drawn from the stationary weights (free target p_T, other-trigger
    outputs 2 p_T, rest p_T, other triggers p_T), normalised. Returns
    (free target count (num_rows,), non-target counts (num_rows, V-K-1))."""
    in_units_of_p_T = [1.0] + [2.0] * (K - 1) + [1.0] * (V - 2 * K) + [K - 1.0]
    weights = torch.tensor(in_units_of_p_T, dtype=dtype, device=mu.device)
    available = (mu - 2 - 2 * ell).clamp(min=0)
    slots = max(int(available.max().item()), 1)
    draws = torch.multinomial(weights, len(mu) * slots, replacement=True,
                              generator=generator).view(len(mu), slots)
    is_real = (torch.arange(slots, device=mu.device) < available[:, None]).to(dtype)
    counts = torch.zeros(len(mu), len(weights), dtype=dtype, device=mu.device).scatter_add_(1, draws, is_real)
    return counts[:, 0], counts[:, 1:-1]


def sample_variables(mu, ell, V: int, K: int, num_samples: int | None = None, counts: str = "poisson",
                     generator: torch.Generator | None = None, dtype=torch.float64, device=None,
                     chunk: int = 4096) -> QueryTable:
    """The variables drawn from their law at fixed (mu, ell), following the
    sampling protocol of the scratch file:

    1. the earlier queries at x_1..x_ell ~ U(0, 1): R = mu sum (1 - x_k);
    2. f free targets at y_j ~ U(0, 1): F = f, W = mu (sum x_k + sum y_j),
       P = ell (ell - 1)/2 + #{(k, j): x_k < y_j};
    3. c_b occurrences of each non-target token b at z ~ U(0, 1): U_bar = c_b,
       W_bar = mu sum z, P_bar = #{(k, i): x_k < z_i}.

    mu, ell: ints or 1-D tensors (one per row), broadcast to `num_samples` rows.
    counts: "poisson" draws f ~ Poisson(m) and c_b ~ Poisson(r_b) independently;
    "multinomial" draws them jointly on the mu - 2 - 2 ell available keys, which
    keeps their weak anti-correlation (matters for sums over many tokens).
    Rows are drawn `chunk` at a time; pass a `torch.Generator` for reproducibility.
    The key columns place every occurrence at a uniform key of the query, the
    witness keys at the same relative positions x_k as R, W and P.
    """
    if counts not in COUNT_LAWS:
        raise ValueError(f"counts must be one of {COUNT_LAWS}, got {counts!r}")
    mu, ell = _as_rows(mu, ell, num_samples, device)
    parts = []
    for start in range(0, len(mu), chunk):
        parts.append(_sample_chunk(mu[start:start + chunk], ell[start:start + chunk], V, K, counts, generator, dtype))
    return _concatenate_padded(parts)


def _to_keys(positions: torch.Tensor, is_real: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Positions in (0, 1) of a query at mu -> uniform keys, code positions
    1..mu-2 (0 where not real or mu has no keys). A witness key is the key after
    the earlier a at the same relative position."""
    n_keys = (mu - 2).clamp(min=0).view(-1, *([1] * (positions.dim() - 1)))
    keys = 1 + (torch.where(is_real, positions, 0.0) * n_keys).long().clamp(max=n_keys - 1)
    return torch.where(is_real & (n_keys > 0), keys, 0)


def _sample_chunk(mu, ell, V, K, counts, generator, dtype) -> QueryTable:
    num_rows = len(mu)
    mu_float, ell_float = mu.to(dtype), ell.to(dtype)
    m = mu_float / (V + K)                                              # p_T mu
    rates = _non_target_rates(m, K, V)                                  # r_b

    # 1. the earlier queries
    query_positions, is_query = _uniform_positions(ell, generator, dtype)
    query_positions = torch.where(is_query, query_positions, torch.inf).sort(dim=1).values
    real_query_positions = torch.where(is_query, query_positions, 0.0)
    R = mu_float * (is_query.to(dtype) - real_query_positions).sum(1)

    # the counts of free targets and of the non-target tokens
    if counts == "poisson":
        free_targets = torch.poisson(m, generator=generator)
        occurrences = torch.poisson(rates, generator=generator)
    else:
        free_targets, occurrences = _multinomial_counts(mu, ell, K, V, generator, dtype)

    # 2. the target
    target_positions, is_target = _uniform_positions(free_targets.long(), generator, dtype)
    target_pairs = (_earlier_queries_behind(query_positions, target_positions) * is_target).sum(1)
    W = mu_float * (real_query_positions.sum(1) + (target_positions * is_target).sum(1))
    P = ell_float * (ell_float - 1) / 2 + target_pairs

    # 3. the non-target tokens
    token_positions, is_token = _uniform_positions(occurrences.long(), generator, dtype)
    W_bar = mu_float[:, None] * (token_positions * is_token).sum(-1)
    P_bar = (_earlier_queries_behind(query_positions, token_positions) * is_token).sum(-1)
    # each occurrence of an output follows its trigger with probability 1/2: Y_bar is the mean of
    # c_tr (c_tr - 1) given the count, so no draw is added and the other variables keep their stream
    is_output = torch.arange(V - K - 1, device=mu.device) < K - 1
    Y_bar = occurrences * (occurrences - 1) / 4 * is_output

    return QueryTable({"mu": mu, "ell": ell, "N": ell_float, "F": free_targets, "R": R, "W": W, "P": P,
                       "U_bar": occurrences, "W_bar": W_bar, "P_bar": P_bar, "Y_bar": Y_bar,
                       "N_keys": _to_keys(query_positions, is_query, mu),
                       "F_keys": _to_keys(target_positions, is_target, mu),
                       "U_keys": _to_keys(token_positions, is_token, mu), "K": K})


def mean_variables(mu, ell, V: int, K: int, dtype=torch.float64, device=None) -> QueryTable:
    """The means of `sample_variables` at fixed (mu, ell) (Table 2), one row per
    (mu, ell) pair (ints or 1-D tensors, broadcast). Since the ansatz logits are
    linear in the variables, `ansatz_logits(mean_variables(...), ...)` is the
    mean logit vector."""
    mu, ell = _as_rows(mu, ell, None, device)
    mu_float, ell_float = mu.to(dtype), ell.to(dtype)
    m = mu_float / (V + K)
    rates = _non_target_rates(m, K, V)
    return QueryTable({
        "mu": mu, "ell": ell, "N": ell_float, "F": m,
        "R": mu_float * ell_float / 2,
        "W": mu_float * (ell_float + m) / 2,
        "P": ell_float * (ell_float - 1) / 2 + m * ell_float / 2,
        "U_bar": rates,
        "W_bar": mu_float[:, None] * rates / 2,
        "P_bar": rates * ell_float[:, None] / 2,
        "Y_bar": (rates / 2) ** 2 * (torch.arange(V - K - 1, device=mu.device) < K - 1),
        "K": K,
    })
