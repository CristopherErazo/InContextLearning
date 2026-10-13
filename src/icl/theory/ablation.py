"""Ablation of the logit noise by mechanism (thread noise-ablation, journal D31): groups of
terms chosen by their tags, `keep=` predicates that drop a group from the full model or add
groups to the signal-only one, and the exact split of the gradient of the effective loss
(method="mean") into the means and one part per group.

    loss = EffectiveLoss(V, L, K, beta, mus=mus, keep=keep_drop("A_tok"))   # full minus a group
    loss = EffectiveLoss(V, L, K, beta, mus=mus, keep=keep_add("R_own"))    # signal plus a group
    split = gradient_split(EffectiveLoss(V, L, K, beta, mus=mus), order_params)
    split["R_own"]["M_on"]                                                  # that group's part

Groups follow the tags of `terms.Term`, never single terms, since terms cancel in pairs.
Cut "A" (by mechanism) and cut "B" (by class) each partition every term that is not a mean:

- A: R_own, R_other, R_trig (trigger-query readout noise by column: own, other columns, the
  trigger rows); A_inc, A_src, A_tok, A_ainc, A_acoh (trigger-query attention noise by
  mechanism); SPREAD (spread of the means given ell, eq. cov_total); NT (the non-trigger
  noise, L_N - log V).
- B: target ("on,on"), triggers ("T,T", "T,T'"), non_targets ("b,b"), target_trigger
  ("on,T"): the trigger-query noise by class; SPREAD and NT as in A.

The split: with the moments X (mean logits and the five closure moments, all sums of terms)
held as leaves, the trigger loss has the adjoint a_X = d loss / d X; each X is linear in
the terms, so d loss / d theta = sum over parts P of d/d theta sum_X <a_X, X(P)>, the parts
being the means and the groups. The non-trigger part is linear in the terms already.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .ansatz import ORDER_PARAMS, PROFILE, ansatz_logits
from .effective import VARIANCE_KEYS, _has_noise, closure_cross_entropy
from .noise import CLOSURE_PAIRS, closure_variances, expand_classes, mean_logits
from .terms import nontrigger_second_order, nontrigger_terms_mu, trigger_terms_ell
from .terms.table import NONTRIGGER_LABELS

NOISE = ("readout", "attention")


@dataclass(frozen=True)
class Select:
    """A predicate on `terms.Term`: the query kind ("trigger", "nontrigger"), the channels
    (empty: any) and, if `tag` is given, the allowed values of that tag."""
    query: str
    channels: tuple = ()
    tag: str = ""
    values: tuple = ()

    def __call__(self, term) -> bool:
        if (term.label in NONTRIGGER_LABELS) != (self.query == "nontrigger"):
            return False
        if self.channels and term.channel not in self.channels:
            return False
        return not self.tag or getattr(term, self.tag) in self.values


MEANS = Select("trigger", ("mean",))
SPREAD = Select("trigger", ("spread",))
NT = Select("nontrigger")
CUTS = {
    "A": {"R_own": Select("trigger", ("readout",), "column", ("own",)),
          "R_other": Select("trigger", ("readout",), "column", ("other",)),
          "R_trig": Select("trigger", ("readout",), "column", ("all",)),
          "A_inc": Select("trigger", ("attention",), "mechanism", ("incoherent",)),
          "A_src": Select("trigger", ("attention",), "mechanism", ("source_coherent",)),
          "A_tok": Select("trigger", ("attention",), "mechanism", ("token_coherent",)),
          "A_ainc": Select("trigger", ("attention",), "mechanism", ("a_incoherent",)),
          "A_acoh": Select("trigger", ("attention",), "mechanism", ("a_coherent",)),
          "SPREAD": SPREAD, "NT": NT},
    "B": {"target": Select("trigger", NOISE, "pair", ("on,on",)),
          "triggers": Select("trigger", NOISE, "pair", ("T,T", "T,T'")),
          "non_targets": Select("trigger", NOISE, "pair", ("b,b",)),
          "target_trigger": Select("trigger", NOISE, "pair", ("on,T",)),
          "SPREAD": SPREAD, "NT": NT},
}
GROUPS = {name: group for cut in CUTS.values() for name, group in cut.items()}


@dataclass(frozen=True)
class Keep:
    """A `keep=` predicate: every term but the `groups` (drop=True), or the means plus the
    `groups` (drop=False)."""
    groups: tuple
    drop: bool

    def __call__(self, term) -> bool:
        hit = any(GROUPS[name](term) for name in self.groups)
        return not hit if self.drop else MEANS(term) or hit


def keep_drop(*names: str) -> Keep:
    """The full model minus the named groups (of `GROUPS`)."""
    return Keep(_known(names), drop=True)


def keep_add(*names: str) -> Keep:
    """The signal-only model (the means, no spread, no noise) plus the named groups."""
    return Keep(_known(names), drop=False)


def _known(names) -> tuple:
    unknown = set(names) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown groups {sorted(unknown)}; groups: {list(GROUPS)}")
    return tuple(names)


def gradient_split(loss, order_params: dict, cut: str = "A") -> dict:
    """{part: {name: d loss / d name}} at `order_params` for an EffectiveLoss with method="mean",
    over the six means and the variances given (no profile). Parts: "total" (autograd of the
    loss), "means" (through the mean logits, with the closure's weights at the full noise),
    one per group of CUTS[cut], so that "means" + the groups = "total"; and "signal", the
    gradient of the signal-only loss (the means alone, CE at the mean logits), for reference.
    The loss's own `keep` is applied first (a group outside it contributes 0)."""
    if loss.method != "mean":
        raise ValueError("gradient_split needs method='mean'")
    if PROFILE in order_params:
        raise ValueError("gradient_split takes the scalar M_on, not a profile")
    if not (_has_noise(order_params) or loss.spread == "always"):
        raise ValueError("without variances and with spread != 'always' the loss is not the closure")
    params = {name: torch.tensor(float(order_params.get(name, 0.0)), dtype=loss.dtype, requires_grad=True)
              for name in (*ORDER_PARAMS, *(name for name in VARIANCE_KEYS if name in order_params))}
    leaves = list(params.values())

    def gradient(value) -> dict:
        if not torch.is_tensor(value) or not value.requires_grad:
            return {name: 0.0 for name in params}
        grads = torch.autograd.grad(value, leaves, retain_graph=True, allow_unused=True)
        return {name: 0.0 if grad is None else grad.item() for name, grad in zip(params, grads)}

    device = loss.variables["N"].device
    q = loss.trigger_probability
    weights = loss.clusters["probability"].to(device) * q / len(loss.mus)
    table = trigger_terms_ell(loss.clusters["mu"], loss.clusters["ell"], params, loss.beta, loss.L, loss.V, loss.K,
                              Lambda=loss.Lambda, dtype=loss.dtype, device=device, pairs=("on", "T", "b", *CLOSURE_PAIRS),
                              classes_only=True)
    kept = {name for name in table.resolve(loss.keep) if loss.spread or table[name].channel != "spread"}

    means = {name for name in table.terms if MEANS(table[name])}
    # the loss's own mean logits: ansatz_logits of the Table 2 means unless keep or Lambda make it the
    # term layer's (the two differ at mu <= 2 when the means are large)
    logits = (mean_logits(table, kept & means) if loss._custom_means()
              else ansatz_logits(loss.variables, params, loss.beta, loss.L))

    def trigger_part(values: dict) -> torch.Tensor:
        variances = {**values, "non_target": expand_classes(values["non_target"], loss.K, loss.V)}
        return (weights * closure_cross_entropy(values["logits"], variances, loss.K)).sum()

    full = {key: value.detach().requires_grad_(True)
            for key, value in {"logits": logits, **closure_variances(table, kept)}.items()}
    adjoint = dict(zip(full, torch.autograd.grad(trigger_part(full), list(full.values()))))

    def linear(names) -> torch.Tensor:
        """<adjoint, moments of the terms `names`>; the logits enter through the means part only."""
        value = sum((adjoint[key] * moment).sum() for key, moment in closure_variances(table, names).items())
        return value + ((adjoint["logits"] * logits).sum() if names & means else 0.0)

    noisy = _has_noise(order_params)
    if noisy:
        nt_table = nontrigger_terms_mu(loss.mus, params, loss.beta, loss.L, loss.V, loss.K, Lambda="n-2ell",
                                       dtype=loss.dtype, device=device)          # as noise.non_trigger_loss
        nt_kept = nt_table.resolve(loss.keep)

    def nontrigger_part(group) -> torch.Tensor | float:
        if not noisy:
            return 0.0
        names = {name for name in nt_kept if group(nt_table[name])}
        return (1 - q) * nontrigger_second_order(nt_table, names).mean() if names else 0.0

    def part(group):
        return linear({name for name in kept if group(table[name])}) + nontrigger_part(group)

    split = {"total": gradient(loss(params)), "means": gradient(part(MEANS))}
    split |= {name: gradient(part(group)) for name, group in CUTS[cut].items()}
    signal = mean_logits(table, means) if loss._custom_means() else logits
    split["signal"] = gradient((weights * (torch.logsumexp(signal, dim=1) - signal[:, -1])).sum())
    return split


def negative_moments(loss, order_params: dict, keep, tol: float = 1e-9) -> dict:
    """{moment: probability mass of the trigger clusters where it is negative} for the closure
    moments of `keep` at `order_params` (the closure counts a negative variance as 0), the mass
    being P(ell | mu) averaged over the mus. Negative = below -tol x the largest |moment| over the
    clusters (the moments that vanish, e.g. at mu <= 2, come out at -1e-13)."""
    with torch.no_grad():
        device = loss.variables["N"].device
        table = trigger_terms_ell(loss.clusters["mu"], loss.clusters["ell"], order_params, loss.beta, loss.L, loss.V,
                                  loss.K, Lambda=loss.Lambda, dtype=loss.dtype, device=device, pairs=CLOSURE_PAIRS,
                                  classes_only=True)
        variances = closure_variances(table, keep)
        mass = loss.clusters["probability"].to(device) / len(loss.mus)
        return {key: float((mass * (value.reshape(len(mass), -1) < -tol * value.abs().max()).any(1)).sum())
                for key, value in variances.items() if key != "target_trigger"}
