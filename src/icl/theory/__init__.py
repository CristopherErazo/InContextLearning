"""The effective model of the logits (archive/scratch/extended_ansatz.tex).

- `query_table`: the `QueryTable` (one row per trigger query), the canonical
  logit layout, the model's logits as a table and their per-(mu, ell)
  cross-entropy. Ansatz-independent.
- `ansatz`: the order-parameter and block-variance registries, the levels, and the ansatz logits.
- `variables`: the sequence variables the ansatz logits depend on.
- `noise`: the fluctuation of the logits in the variance ansatz (variances, sampled
  logits, the non-trigger loss).
- `effective`: the effective population loss, differentiable in the order parameters,
  and its gradient flow (`integrate`).
- `terms`: the logit moments of the appendix as named, prunable terms at every level
  of conditioning (exact given the sequence, given S_mu, given ell / mu).

Only `ansatz`, `variables` and `noise` know the ansatz; see `ansatz` for how to extend it.
"""
from .query_table import (QueryTable, canonical_permutation, check_condition, cluster_cross_entropy,
                          condition_mask, logit_blocks, logit_table, trigger_queries)
from .ansatz import (DIAGNOSTIC_MEANS, LEVELS, ORDER_PARAMS, POOLED_VARIANCES, PROFILE, SIGNAL, VARIANCES, ansatz_logits,
                     ansatz_matrices, at_level, block_variance, measure_diagnostic_means, measure_order_params,
                     measure_variances, support_sizes, variance_sizes)
from .variables import (COUNT_LAWS, mean_variables, measure_nontrigger_variables, measure_variables,
                        nontrigger_blocks, nontrigger_permutation, nontrigger_queries, sample_nontrigger_variables,
                        sample_variables)
from .noise import noise_variances, non_trigger_loss, sample_logits
from . import terms
from .terms import (TermTable, exact_nontrigger_terms, exact_trigger_terms, nontrigger_terms, nontrigger_terms_mu,
                    trigger_terms, trigger_terms_ell)
from .effective import (EffectiveLoss, asymptotic_loss, closure_cross_entropy, exact_rates, integrate, learning_time,
                        trigger_loss)

__all__ = [
    "QueryTable", "canonical_permutation", "cluster_cross_entropy", "logit_blocks", "logit_table",
    "trigger_queries", "check_condition", "condition_mask",
    "ORDER_PARAMS", "PROFILE", "ansatz_logits", "ansatz_matrices", "measure_order_params", "support_sizes",
    "DIAGNOSTIC_MEANS", "VARIANCES", "POOLED_VARIANCES", "SIGNAL", "LEVELS", "at_level", "block_variance",
    "measure_diagnostic_means", "measure_variances", "variance_sizes",
    "COUNT_LAWS", "mean_variables", "measure_variables", "sample_variables", "measure_nontrigger_variables",
    "sample_nontrigger_variables", "nontrigger_blocks", "nontrigger_permutation", "nontrigger_queries",
    "noise_variances", "non_trigger_loss", "sample_logits",
    "EffectiveLoss", "asymptotic_loss", "closure_cross_entropy", "exact_rates", "integrate", "learning_time",
    "trigger_loss",
    "terms", "TermTable", "trigger_terms", "trigger_terms_ell", "nontrigger_terms", "nontrigger_terms_mu",
    "exact_trigger_terms", "exact_nontrigger_terms",
]
