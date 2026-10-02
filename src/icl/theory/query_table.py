"""The data side of the effective model: a table with one row per trigger query,
the canonical layout of a logit vector, and the model's logits as such a table.

Conventions (shared by every function of `icl.theory`):

- positions are the paper's mu = code position + 1 (the query is tau_mu, the
  keys are nu = 2..mu-1), and ell = number of earlier occurrences of the query;
- only trigger queries make rows, every ell >= 0 included: the ansatz gives
  zero logits at every other query;
- a logit vector is stored in the canonical layout of `logit_blocks`, so the
  target is always the last column.
"""
from __future__ import annotations

import numbers
import re

import torch

from ..data import occurrence_counts

_AT_LEAST = re.compile(r"^>=\s*(\d+)$")


def check_condition(condition) -> None:
    """Raise unless `condition` is None, an int, a non-empty list of ints or '>=k'."""
    def is_integer(value):
        return isinstance(value, numbers.Integral) and not isinstance(value, bool)
    ok = (condition is None or is_integer(condition)
          or (isinstance(condition, (list, tuple)) and condition and all(map(is_integer, condition)))
          or (isinstance(condition, str) and _AT_LEAST.match(condition)))
    if not ok:
        raise ValueError(f"a condition must be None, an int, a list of ints or '>=k', got {condition!r}")


def condition_mask(column: torch.Tensor, condition) -> torch.Tensor:
    """Where `column` matches `condition`: None (everywhere), an int (==), a
    list (in it) or '>=k' (>= k)."""
    check_condition(condition)
    if condition is None:
        return torch.ones_like(column, dtype=torch.bool)
    if isinstance(condition, str):
        return column >= int(_AT_LEAST.match(condition).group(1))
    return torch.isin(column, torch.as_tensor(condition, dtype=column.dtype, device=column.device).reshape(-1))


class QueryTable(dict):
    """A table with one row per trigger query: a dict whose tensor values (the
    columns) all have the number of rows as their first dimension, with at
    least the columns "mu" and "ell". Non-tensor values (e.g. "K") are metadata
    and pass through every operation untouched.

        table["logits"]                          # a column, here (num_rows, V)
        table.num_rows
        table.select(mu=256, ell=[1, 2])         # rows with mu == 256 and ell in {1, 2}
        table.select(ell=">=1")                  # rows with ell >= 1
        for (mu, ell), cluster in table.groups("mu", "ell"): ...
        table.to_numpy() / QueryTable.from_numpy(saved)    # plain dict of numpy arrays
        QueryTable.concatenate([table_1, table_2])
    """

    @property
    def num_rows(self) -> int:
        return len(self["mu"])

    def _columns(self) -> dict[str, torch.Tensor]:
        return {name: value for name, value in self.items()
                if torch.is_tensor(value) and value.dim() and len(value) == self.num_rows}

    def _take(self, row_index) -> "QueryTable":
        columns = self._columns()
        return QueryTable({name: (columns[name][row_index] if name in columns else value)
                           for name, value in self.items()})

    def select(self, **conditions) -> "QueryTable":
        """The rows that match every condition: column=value, column=[values]
        or column=">=k" (see `condition_mask`)."""
        keep = torch.ones(self.num_rows, dtype=torch.bool)
        for name, condition in conditions.items():
            keep &= condition_mask(self[name], condition).cpu()
        return self._take(keep)

    def groups(self, *columns):
        """Iterate over ((value of each column), sub-table) for every distinct
        combination of values of `columns`, in sorted order."""
        values = torch.stack([self[name] for name in columns], dim=1)
        for combination in torch.unique(values, dim=0):
            in_group = (values == combination).all(dim=1)
            yield tuple(value.item() for value in combination), self._take(in_group)

    def to_numpy(self) -> dict:
        return {name: (value.detach().cpu().numpy() if torch.is_tensor(value) else value)
                for name, value in self.items()}

    @classmethod
    def from_numpy(cls, saved: dict) -> "QueryTable":
        return cls({name: (torch.as_tensor(value) if hasattr(value, "shape") else value)
                    for name, value in saved.items()})

    @classmethod
    def concatenate(cls, tables: list["QueryTable"]) -> "QueryTable":
        first = tables[0]
        columns = first._columns()
        return cls({name: (torch.cat([table[name] for table in tables]) if name in columns else value)
                    for name, value in first.items()})


def logit_blocks(K: int, V: int) -> dict[str, slice]:
    """Column slices of the canonical layout of a logit vector at a query a:

        triggers       [0, K)        the K triggers, by token id
        other_outputs  [K, 2K-1)     the outputs phi(c) of the other triggers c != a, by c
        rest           [2K-1, V-1)   the V-2K remaining tokens, by token id
        target         [V-1, V)      the target phi(a)
        non_target     [K, V-1)      other_outputs + rest: every non-trigger but the target
    """
    return {"triggers": slice(0, K), "other_outputs": slice(K, 2 * K - 1),
            "rest": slice(2 * K - 1, V - 1), "target": slice(V - 1, V), "non_target": slice(K, V - 1)}


def canonical_permutation(query_tokens: torch.Tensor, output_sets: torch.Tensor, K: int, V: int) -> torch.Tensor:
    """(num_rows, V) long: entry [row, slot] is the token shown at `slot` of the
    canonical layout, so `logits.gather(1, permutation)` reorders logit vectors.

    query_tokens: (num_rows,) the trigger at each query. output_sets: (num_rows, K)
    the outputs of that row's sequence (trigger c maps to output_sets[row, c]).
    One argsort of a sort key: trigger c -> c, output of a trigger c != a -> V + c,
    any other token t -> 2V + t, target -> 3V.
    """
    num_rows, device = query_tokens.size(0), query_tokens.device
    sort_key = (2 * V + torch.arange(V, device=device)).repeat(num_rows, 1)
    sort_key[:, :K] = torch.arange(K, device=device)
    sort_key.scatter_(1, output_sets, (V + torch.arange(K, device=device)).repeat(num_rows, 1))
    target_tokens = output_sets.gather(1, query_tokens[:, None])
    sort_key.scatter_(1, target_tokens, 3 * V)
    return sort_key.argsort(dim=1)


def trigger_queries(inputs: torch.Tensor, K: int, mus=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(sequence index, code position) of every trigger query of `inputs` (B, L)
    at the given mu (int, list, or None = every mu), in row-major order.
    `logit_table` and `measure_variables` both use it, so on the same batch
    their rows line up one to one."""
    is_query = inputs < K
    if mus is not None:
        position_wanted = torch.zeros(inputs.size(1), dtype=torch.bool, device=inputs.device)
        positions = torch.as_tensor(mus, device=inputs.device).reshape(-1) - 1
        position_wanted[positions[(positions >= 0) & (positions < inputs.size(1))]] = True
        is_query &= position_wanted
    return torch.where(is_query)


def logit_table(model, batch: dict, mus=None, chunk: int | None = None) -> QueryTable:
    """The model's logits at every trigger query of `batch` (at the chosen mu),
    in the canonical layout.

    Returns a QueryTable with "logits" (num_rows, V), "mu", "ell",
    "sequence_index" (index of the sequence in the batch) and the metadata "K",
    on the CPU. Only `sequence`, `output_set` and `K` of the batch are read.
    With pred_mode="last" only mu = L exists. The model runs in eval mode
    without gradients, `chunk` sequences at a time.
    """
    K = int(batch["K"])
    inputs, output_sets = batch["sequence"][:, :-1].cpu(), batch["output_set"].cpu()
    num_sequences, L = inputs.shape
    sequence_index, position = trigger_queries(inputs, K, mus)
    ell = (occurrence_counts(inputs) - 1)[sequence_index, position]
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    kept_rows, kept_logits = [], []
    try:
        with torch.inference_mode():
            chunk = chunk or num_sequences
            for start in range(0, num_sequences, chunk):
                in_chunk = (sequence_index >= start) & (sequence_index < start + chunk)
                logits = model(inputs[start:start + chunk].to(device)).cpu()     # (chunk, Lq, V)
                output_position = position[in_chunk] - (L - logits.size(1))      # 'last' mode keeps only L-1
                emitted = output_position >= 0
                kept_rows.append(in_chunk.nonzero().squeeze(1)[emitted])
                kept_logits.append(logits[sequence_index[in_chunk][emitted] - start, output_position[emitted]])
    finally:
        model.train(was_training)
    rows = torch.cat(kept_rows)
    logits = torch.cat(kept_logits)
    sequence_index, position = sequence_index[rows], position[rows]
    permutation = canonical_permutation(inputs[sequence_index, position], output_sets[sequence_index],
                                        K, logits.size(1))
    return QueryTable({"logits": logits.gather(1, permutation), "mu": position + 1, "ell": ell[rows],
                       "sequence_index": sequence_index, "K": K})


def cluster_cross_entropy(table: QueryTable) -> QueryTable:
    """One row per (mu, ell) cluster of `table`: the number of queries "count"
    and the mean "cross_entropy" of `table["logits"]` against the target (the
    last column). Differentiable in the logits."""
    logits = table["logits"]
    cross_entropy = torch.logsumexp(logits, dim=1) - logits[:, -1]
    clusters, cluster_of_row = torch.unique(torch.stack([table["mu"], table["ell"]], dim=1),
                                            dim=0, return_inverse=True)
    zeros = torch.zeros(len(clusters), dtype=cross_entropy.dtype, device=cross_entropy.device)
    count = zeros.index_add(0, cluster_of_row, torch.ones_like(cross_entropy))
    total = zeros.index_add(0, cluster_of_row, cross_entropy)
    return QueryTable({"mu": clusters[:, 0], "ell": clusters[:, 1], "count": count,
                       "cross_entropy": total / count})
