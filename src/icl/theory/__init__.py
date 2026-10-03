"""The effective model of the logits (paper/scratch/extended_ansatz.tex).

- `query_table`: the `QueryTable` (one row per trigger query), the canonical
  logit layout, the model's logits as a table and their per-(mu, ell)
  cross-entropy. Ansatz-independent.
- `ansatz`: the order-parameter registry and the ansatz logits.
- `variables`: the sequence variables the ansatz logits depend on.
- `effective`: the effective population loss, differentiable in the order parameters,
  and its gradient flow (`integrate`).

Only `ansatz` and `variables` know the ansatz; see `ansatz` for how to extend it.
"""
from .query_table import (QueryTable, canonical_permutation, check_condition, cluster_cross_entropy,
                          condition_mask, logit_blocks, logit_table, trigger_queries)
from .ansatz import ORDER_PARAMS, PROFILE, ansatz_logits, ansatz_matrices, measure_order_params, support_sizes
from .variables import COUNT_LAWS, mean_variables, measure_variables, sample_variables
from .effective import EffectiveLoss, integrate, trigger_loss

__all__ = [
    "QueryTable", "canonical_permutation", "cluster_cross_entropy", "logit_blocks", "logit_table",
    "trigger_queries", "check_condition", "condition_mask",
    "ORDER_PARAMS", "PROFILE", "ansatz_logits", "ansatz_matrices", "measure_order_params", "support_sizes",
    "COUNT_LAWS", "mean_variables", "measure_variables", "sample_variables",
    "EffectiveLoss", "integrate", "trigger_loss",
]
