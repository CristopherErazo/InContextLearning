"""The noise ablation (`icl.theory.ablation`): each cut partitions the terms that are not
means at both kinds of query; the gradient split adds up to the autograd gradient; adding
no group is the signal-only loss and adding every group is the full loss."""
from __future__ import annotations

import math

import pytest
import torch

from icl import EffectiveLoss, nontrigger_terms_mu, trigger_terms_ell
from icl.theory.ablation import CUTS, MEANS, gradient_split, keep_add, keep_drop, negative_moments
from icl.theory.noise import CLOSURE_PAIRS

V, L, K, BETA = 48, 96, 9, 0.25
MEANS_OP = {"M_on": 0.5, "M_off": 0.02, "Q_on": 0.8, "Q_T": -0.05, "G_on": 0.6, "G_T": -0.1}
SPREAD = {"var_M_on": 0.3, "var_M_off": 0.04, "var_Q_on": 0.5, "var_Q_T": 0.2, "var_Q_N": 0.2, "var_G_on": 0.4,
          "var_G_T": 0.2, "var_G_N": 0.2}
MUS = range(8, L + 1, 8)


def tables():
    op = {**MEANS_OP, **SPREAD}
    trigger = trigger_terms_ell(torch.tensor([24, 96]), torch.tensor([0, 2]), op, BETA, L, V, K,
                                pairs=("on", "T", "b", *CLOSURE_PAIRS), classes_only=True)
    return trigger, nontrigger_terms_mu(torch.tensor([24, 96]), op, BETA, L, V, K)


@pytest.mark.parametrize("cut", list(CUTS))
def test_each_cut_partitions_the_terms_that_are_not_means(cut):
    for table in tables():
        for term in table:
            hits = [name for name, group in CUTS[cut].items() if group(term)]
            assert len(hits) == (0 if MEANS(term) else 1), (term.name, hits)


LEARNED = {"M_on": -5.3, "M_off": -0.34, "Q_on": -7.0, "Q_T": 0.38, "G_on": 3.4, "G_T": 0.31}


@pytest.mark.parametrize("cut", list(CUTS))
@pytest.mark.parametrize("variances", [True, False])
@pytest.mark.parametrize("keep", [None, keep_drop()])
@pytest.mark.parametrize("means, mus", [(MEANS_OP, MUS), (LEARNED, range(1, L + 1, 5))])
def test_the_split_adds_up_to_the_gradient(cut, variances, keep, means, mus):
    """With variances; and without, where the spread of the means (spread="always") is the
    only non-mean part. Also with large means and mu = 1, where the mean logits of keep=None
    (Table 2 means) and of a keep (the term layer) differ."""
    op = {**means, **(SPREAD if variances else {})}
    loss = EffectiveLoss(V, L, K, BETA, mus=mus, keep=keep)
    split = gradient_split(loss, op, cut)
    _, reference = loss.value_and_grad(op)
    for name, total in split["total"].items():
        assert total == pytest.approx(reference[name], rel=1e-12, abs=1e-15)
        parts = split["means"][name] + sum(split[group][name] for group in CUTS[cut])
        assert parts == pytest.approx(total, rel=1e-10, abs=1e-12 * max(map(abs, reference.values()))), name
    if not variances:
        assert all(split[group][name] == 0.0 for group in CUTS[cut] if group != "SPREAD" for name in split["total"])


def test_the_signal_part_is_the_signal_only_gradient():
    op = {**MEANS_OP, **SPREAD}
    split = gradient_split(EffectiveLoss(V, L, K, BETA, mus=MUS), op)
    _, signal = EffectiveLoss(V, L, K, BETA, mus=MUS, spread=False).value_and_grad(MEANS_OP)
    for name, value in signal.items():
        assert split["signal"][name] == pytest.approx(value, rel=1e-10, abs=1e-15)


def test_adding_nothing_is_signal_only_and_adding_everything_is_full():
    op = {**MEANS_OP, **SPREAD}
    signal = EffectiveLoss(V, L, K, BETA, mus=MUS, spread=False)(MEANS_OP)
    assert EffectiveLoss(V, L, K, BETA, mus=MUS, keep=keep_add())(op).item() == pytest.approx(signal.item(), rel=1e-13)
    full = EffectiveLoss(V, L, K, BETA, mus=MUS)(op)
    for cut in CUTS.values():
        assert EffectiveLoss(V, L, K, BETA, mus=MUS, keep=keep_add(*cut))(op).item() == pytest.approx(full.item(),
                                                                                                    rel=1e-13)
        assert EffectiveLoss(V, L, K, BETA, mus=MUS, keep=keep_drop(*cut))(op).item() == pytest.approx(signal.item(),
                                                                                                     rel=1e-13)


def test_dropping_the_non_trigger_noise_leaves_log_V():
    loss = EffectiveLoss(V, L, K, BETA, mus=MUS, keep=keep_drop("NT"))
    assert float(loss.non_trigger_loss({**MEANS_OP, **SPREAD})) == pytest.approx(math.log(V), rel=1e-15)


def test_negative_moments_and_unknown_groups():
    loss = EffectiveLoss(V, L, K, BETA, mus=MUS)
    masses = negative_moments(loss, {**MEANS_OP, **SPREAD}, None)
    assert set(masses) == {"target", "trigger_common", "trigger_individual", "non_target"}
    assert all(0.0 <= mass <= 1.0 + 1e-12 for mass in masses.values())
    with pytest.raises(ValueError):
        keep_drop("nonsense")
