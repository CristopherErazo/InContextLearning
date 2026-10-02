"""The effective model of the logits (paper/scratch/extended_ansatz.tex).

- `query_table`: the `QueryTable` (one row per trigger query), the canonical
  logit layout, the model's logits as a table and their per-(mu, ell)
  cross-entropy. Ansatz-independent.
- `ansatz`: the order-parameter registry and the ansatz logits.
- `variables`: the sequence variables the ansatz logits depend on.

Only `ansatz` and `variables` know the ansatz; see `ansatz` for how to extend it.
"""
from .query_table import (QueryTable, canonical_permutation, cluster_cross_entropy, logit_blocks,
                          logit_table, trigger_queries)
from .ansatz import ORDER_PARAMS, ansatz_logits, ansatz_matrices, measure_order_params, support_sizes
from .variables import measure_variables

__all__ = [
    "QueryTable", "canonical_permutation", "cluster_cross_entropy", "logit_blocks", "logit_table",
    "trigger_queries",
    "ORDER_PARAMS", "ansatz_logits", "ansatz_matrices", "measure_order_params", "support_sizes",
    "measure_variables",
]
