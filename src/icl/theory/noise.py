"""The fluctuation of the logits in the variance ansatz: h = h_bar + xi, h_bar the mean
logits (`ansatz_logits`) and xi the noise of the block fluctuations, with zero mean given
the sequence. Since 2026-10-08 every moment comes from the term layer `icl.theory.terms`
(the appendix's eqs. pairs_trigger, pairs_nontrigger_mean, with the corrected
token-coherent rule and without the same-source pairs, audit items a1, b1, b2, c1 of
paper/scratch/2026-10-08-1119_pruned-effective-model.tex):

    noise_variances(variables, order_params, beta, L)         # per trigger query, given S_mu
    sample_logits(variables, order_params, beta, L, generator) # h_bar + a Gaussian xi
    non_trigger_loss(mu, order_params, beta, L, V, K)          # L_N(mu) to second order

All take `keep=` (term names, a predicate on the terms or a preset of
`icl.theory.terms.PRESETS`) to prune the terms. A missing variance counts as 0: without
any, every variance is exactly 0, `sample_logits` is `ansatz_logits` and
`non_trigger_loss` is log V.
"""
from __future__ import annotations

import math

import torch

from .ansatz import ansatz_logits
from .query_table import QueryTable, logit_blocks
from .sums import _diagonal, _key_sums, _parameters, _witness_sums  # noqa: F401  (re-exported)
from .terms import nontrigger_second_order, nontrigger_terms_mu, trigger_terms

CLOSURE_PAIRS = ("on,on", "T,T", "T,T'", "b,b", "on,T")
COVARIANCE_PAIRS = ("on,b", "b,T", "b,b'")


def closure_variances(table, keep=None) -> dict:
    """The five moments the closure and `sample_logits` use, from a trigger-query TermTable
    (any level): "target" Var h^on, "trigger_common" V_c^T = Cov(h_tau, h_tau'),
    "trigger_individual" V_i^T = Var h_tau - V_c^T, "non_target" Var h_b, "target_trigger"
    Cov(h^on, h_tau). Given ell they include the spread of the means unless `keep` drops it."""
    kept = table.resolve(keep)
    # the attention and spread terms of (T, T) are those of (T, T'): what is individual is the readout
    individual = {name for name in kept if table[name].pair == "T,T" and table[name].channel == "readout"}
    return {"target": table.total("on,on", kept), "trigger_common": table.total("T,T'", kept),
            "trigger_individual": table.total("T,T", individual), "non_target": table.total("b,b", kept),
            "target_trigger": table.total("on,T", kept)}


def noise_variances(variables: QueryTable, order_params: dict, beta: float, L: int, keep=None,
                    covariances: bool = False) -> dict:
    """The covariance of the logit fluctuation xi at the trigger queries of `variables`
    (measured or sampled: the key columns are needed), given S_mu (eq. pairs_trigger):

        "target"              Var h^on (num_rows,)
        "trigger_common"      V_c^T, the part shared by the K triggers
        "trigger_individual"  V_i^T, i.i.d. over the triggers
        "non_target"          Var h_b (num_rows, V-K-1), in the `non_target` block order
        "target_trigger"      Cov(h^on, h_tau)

    and with covariances=True also "target_non_target" Cov(h^on, h_b) and
    "non_target_trigger" Cov(h_b, h_tau) (num_rows, V-K-1) and "non_target_pairs"
    Cov(h_b, h_b') (a `terms.Bilinear`), which the closure does not use. The predecessor
    type of the keys ("U_triggered") is used if present, else averaged given the keys
    (`terms.trigger_statistics`). Differentiable in the order parameters, the profile and the
    variances. Their mean given (mu, ell) is `terms.trigger_terms_ell`."""
    pairs = CLOSURE_PAIRS + (COVARIANCE_PAIRS if covariances else ())
    table = trigger_terms(variables, order_params, beta, L, pairs=pairs)
    out = closure_variances(table, keep)
    if covariances:
        out.update(target_non_target=table.total("on,b", keep), non_target_trigger=table.total("b,T", keep),
                   non_target_pairs=table.total("b,b'", keep))
    return out


def _sqrt(variance: torch.Tensor) -> torch.Tensor:
    """The standard deviation, 0 (with a zero gradient) where the variance is not positive."""
    positive = variance > 0
    return torch.where(positive, variance, 1).sqrt() * positive


def _target_and_common(target, common, covariance, target_normal, common_normal) -> tuple[torch.Tensor, torch.Tensor]:
    """(xi_on, xi_c^T) from two standard normals, through the Cholesky factor of their
    covariance [[target, covariance], [covariance, common]] (broadcast); what is not
    positive counts as 0."""
    sd_target = _sqrt(target)
    has_target = sd_target > 0
    loading = covariance / torch.where(has_target, sd_target, 1) * has_target
    return sd_target * target_normal, loading * target_normal + _sqrt(common - loading ** 2) * common_normal


def sample_logits(variables: QueryTable, order_params: dict, beta: float, L: int,
                  generator: torch.Generator | None = None, normals: torch.Tensor | None = None,
                  keep=None) -> torch.Tensor:
    """(num_rows, V) logits in the canonical layout: the mean logits plus a Gaussian draw
    of the fluctuation with the moments of `noise_variances`: the target and the common
    trigger part jointly (with their covariance), the individual trigger parts i.i.d., the
    non-targets independent.

    `normals`: (num_rows, V + 1) standard normals, one per entry of the canonical layout
    and the common trigger part last; fixed normals are common random numbers (the
    result is then smooth in the order parameters). Drawn with `generator` if not given.
    keep: prunes the noise terms and, if given, the mean terms (then the means are the
    kept terms of eq. mean_logits). The formulas are leading order: a negative variance
    counts as 0."""
    if keep is None:
        logits = ansatz_logits(variables, order_params, beta, L)
        variances = noise_variances(variables, order_params, beta, L)
    else:
        table = trigger_terms(variables, order_params, beta, L, pairs=("on", "T", "b", *CLOSURE_PAIRS))
        variances = closure_variances(table, keep)
        logits = mean_logits(table, keep)
    num_rows, V = logits.shape
    blocks = logit_blocks(int(variables["K"]), V)
    if normals is None:
        normals = torch.randn(num_rows, V + 1, generator=generator, dtype=logits.dtype, device=logits.device)
    target, common = _target_and_common(variances["target"], variances["trigger_common"], variances["target_trigger"],
                                        normals[:, blocks["target"]].squeeze(1), normals[:, -1])
    triggers = common[:, None] + _sqrt(variances["trigger_individual"])[:, None] * normals[:, blocks["triggers"]]
    non_target = _sqrt(variances["non_target"]) * normals[:, blocks["non_target"]]
    return logits + torch.cat([triggers, non_target, target[:, None]], dim=1)


def mean_logits(table, keep=None) -> torch.Tensor:
    """(rows, V) mean logits in the canonical layout from the kept mean terms of a
    trigger-query TermTable (a classes_only table is expanded to the V-K-1 non-targets)."""
    K, V = table.K, table.V
    b = table.total("b", keep)
    if b.size(1) != V - K - 1:
        b = expand_classes(b, K, V)
    return torch.cat([table.total("T", keep)[:, None].expand(-1, K), b, table.total("on", keep)[:, None]], dim=1)


def expand_classes(value: torch.Tensor, K: int, V: int) -> torch.Tensor:
    """(rows, 2) per class (outputs, plain) -> (rows, V-K-1) in the `non_target` block order."""
    index = torch.cat([torch.zeros(K - 1, dtype=torch.long), torch.ones(V - 2 * K, dtype=torch.long)]).to(value.device)
    return value[:, index]


def non_trigger_loss(mu, order_params: dict, beta: float, L: int, V: int, K: int,
                     dtype=torch.float64, device=None, keep=None, Lambda: str = "n-2ell") -> torch.Tensor:
    """The expected cross-entropy at a non-trigger query at position mu (int or 1-D), to
    second order in the fluctuation: log V + (1/2)[(1/V) sum_c Var xi_c - Var xi_bar], with
    the moments of eq. pairs_nontrigger_mean (`terms.nontrigger_terms_mu`), every pair of
    classes included. Lambda: the occurrence rate of a token, "n-2ell" = p_T (mu - 2) (the
    mu - 2 keys; audit item c1) or "mu" = p_T mu. Exactly log V without variances."""
    table = nontrigger_terms_mu(mu, order_params, beta, L, V, K, Lambda=Lambda, dtype=dtype, device=device)
    return math.log(V) + nontrigger_second_order(table, keep)
