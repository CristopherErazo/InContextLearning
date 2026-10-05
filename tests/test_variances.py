"""The block variances of the variance ansatz: the estimators recover known
variances from `ansatz_matrices(..., noise=generator)` draws (the sub-diagonal
around a smooth profile included), the pooled variances, the levels, and the
order-parameter probes."""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from icl import (DIAGNOSTIC_MEANS, LEVELS, ORDER_PARAMS, POOLED_VARIANCES, VARIANCES, BlockVariances, Evaluator,
                 MProfile, OrderParameters, ReducedTransformer, ansatz_matrices, at_level, block_variance,
                 generate_icl_batch, measure_diagnostic_means, measure_order_params, measure_variances,
                 order_parameter_probes, preprocess_batch, variance_sizes)


@dataclass
class Args:
    vocab_size: int = 128
    seq_len: int = 256
    lin_attn: bool = True
    beta: float = 0.25
    pred_mode: str = "next"
    dropout: float = 0.0
    mask1: str = "causal"
    mask2: str = "causal"


V, L, K = 128, 256, 25
MEANS = {"M_on": 1.0, "M_off": 0.1, "Q_on": 3.0, "Q_T": 0.5, "G_on": 2.0, "G_T": -0.7}
BLOCK_VARIANCES = {"var_M_on": 0.25, "var_M_off": 0.49, "var_Q_on": 0.64, "var_Q_T": 0.36, "var_Q_N": 0.16,
                   "var_G_on": 0.25, "var_G_T": 0.36, "var_G_N": 0.09}


def draw(order_params, seed=0):
    return ansatz_matrices(order_params, L, V, K, noise=torch.Generator().manual_seed(seed))


def test_block_variances_are_recovered():
    matrices = draw({**MEANS, **BLOCK_VARIANCES})
    measured, sizes = measure_variances(matrices, K), variance_sizes(L, V, K)
    for name, variance in BLOCK_VARIANCES.items():
        # sd of a variance estimate over n entries: sqrt(2/n); 3/n for the neighbouring differences
        sd = variance * ((3 if name == "var_M_on" else 2) / sizes[name]) ** 0.5
        assert abs(measured[name] - variance) < 5 * sd, name
    means = measure_order_params(matrices, K)
    support = variance_sizes(L, V, K)
    for name, mean in MEANS.items():
        assert abs(means[name] - mean) < 5 * (BLOCK_VARIANCES["var_" + name] / support["var_" + name]) ** 0.5, name
    for name, mean in measure_diagnostic_means(matrices, K).items():         # zero in the ansatz
        assert abs(mean) < 5 * (BLOCK_VARIANCES["var_" + name] / support["var_" + name]) ** 0.5, name
    assert set(measured) == {*VARIANCES, *POOLED_VARIANCES, "var_M_on_raw"}
    assert set(DIAGNOSTIC_MEANS) == {"Q_N", "G_N"}


def test_the_sub_diagonal_spread_is_around_the_smooth_profile():
    profile = MEANS["M_on"] * (1.6 - 1.2 * torch.arange(2, L + 1, dtype=torch.float64) / L) * 3
    matrices = draw({**MEANS, "M_profile": profile, "var_M_on": 0.25})
    measured = measure_variances(matrices, K)
    assert measured["var_M_on"] == pytest.approx(0.25, rel=5 * (3 / (L - 1)) ** 0.5)
    raw = 0.25 + profile.var(unbiased=False).item()                         # the profile counts as spread
    assert measured["var_M_on_raw"] == pytest.approx(raw, rel=5 * (2 / (L - 1)) ** 0.5)
    assert measured["var_M_on_raw"] > 3 * measured["var_M_on"]


def test_pooled_variances():
    matrices = draw({**MEANS, **BLOCK_VARIANCES}, seed=1)
    measured, sizes = measure_variances(matrices, K), variance_sizes(L, V, K)
    for pooled, matrix_name in POOLED_VARIANCES.items():
        blocks = [name for name, (block_matrix, _) in VARIANCES.items() if block_matrix == matrix_name]
        within = sum(sizes[name] * measured[name] for name in blocks) / sum(sizes[name] for name in blocks)
        assert measured[pooled] == pytest.approx(within, rel=1e-12)
        assert sizes[pooled] == sum(sizes[name] for name in blocks)
    # the raw variance of a matrix = the pooled within-block one + the spread of the block means
    block_means = {**measure_order_params(matrices, K), **measure_diagnostic_means(matrices, K)}
    Q_blocks = {"Q_on": "var_Q_on", "Q_T": "var_Q_T", "Q_N": "var_Q_N"}
    overall = matrices["Q"].mean().item()
    between = sum(sizes[block] * (block_means[mean] - overall) ** 2 for mean, block in Q_blocks.items()) / V ** 2
    assert matrices["Q"].var(unbiased=False).item() == pytest.approx(measured["var_Q"] + between, rel=1e-10)
    # one pooled key stands in for every block of its matrix without its own key
    pooled_only = draw({**MEANS, "var_Q": 0.5, "var_Q_on": 0.04}, seed=2)
    by_block = measure_variances(pooled_only, K)
    assert by_block["var_Q_T"] == pytest.approx(0.5, rel=0.1) and by_block["var_Q_N"] == pytest.approx(0.5, rel=0.1)
    assert by_block["var_Q_on"] == pytest.approx(0.04, rel=1.5)
    assert by_block["var_M_off"] == pytest.approx(0.0, abs=1e-20) and by_block["var_G_T"] == pytest.approx(0.0, abs=1e-20)
    assert block_variance({"var_Q": 0.5, "var_Q_on": 0.04}, "var_Q_T") == 0.5
    assert block_variance({"var_Q": 0.5, "var_Q_on": 0.04}, "var_Q_on") == 0.04
    assert block_variance({}, "var_G_N") == 0.0


def test_no_variance_no_noise():
    for name, matrix in draw(MEANS).items():
        assert torch.equal(matrix, ansatz_matrices(MEANS, L, V, K)[name])


def test_levels():
    full = {**MEANS, "M_profile": torch.ones(L - 1), **BLOCK_VARIANCES, "var_M": 0.3, "var_Q": 0.2, "var_G": 0.1,
            "var_M_on_raw": 1.0}
    assert set(at_level(full, "S0")) == {*MEANS, "M_profile", *BLOCK_VARIANCES}
    assert set(at_level(full, "S1")) == {*MEANS, "M_profile", *POOLED_VARIANCES}
    assert set(at_level(full, "S2")) == {*MEANS, *POOLED_VARIANCES}
    assert set(at_level(full, "S3")) == set(ORDER_PARAMS)
    assert set(at_level(full, "S4")) == {"M_on", "Q_on", "G_on", *POOLED_VARIANCES}
    assert at_level(full, "S5") == {"M_on": 1.0, "Q_on": 3.0, "G_on": 2.0}
    assert list(LEVELS) == ["S0", "S1", "S2", "S3", "S4", "S5"]
    with pytest.raises(ValueError):
        at_level(full, "S6")


def test_order_parameter_probes():
    order_params = {**MEANS, **BLOCK_VARIANCES, "M_profile": torch.linspace(3, 1, L - 1, dtype=torch.float64)}
    matrices = draw(order_params, seed=3)
    model = ReducedTransformer.from_matrices(matrices, Args())
    test_batch, _ = preprocess_batch(generate_icl_batch(8, V, L, K, generator=torch.Generator().manual_seed(4)))
    scalars, artifacts = order_parameter_probes(["means", "variances", "profile"])
    evaluator = Evaluator(scalars=scalars, artifacts=artifacts)
    metrics = evaluator.scalars(model, test_batch, step=0)
    expected = {**measure_order_params(matrices, K), **measure_diagnostic_means(matrices, K),
                **measure_variances(matrices, K)}
    assert metrics.keys() == expected.keys()
    for name, value in expected.items():
        assert metrics[name] == pytest.approx(value, rel=1e-12), name
    saved = evaluator.artifacts(model, test_batch, step=0, schedule="scalar")[("M_profile", "profile")][0]
    torch.testing.assert_close(torch.as_tensor(saved), matrices["M"].diagonal(-1))
    assert evaluator.artifacts(model, test_batch, step=0) == {}             # not on the artifact schedule
    assert evaluator.artifact_schedules == {"scalar"}
    with pytest.raises(ValueError):
        evaluator.artifacts(model, test_batch, step=0, schedule="never")

    # single names, the default of OrderParameters, and the switches off
    scalars, artifacts = order_parameter_probes(["M_on", "Q_on", "G_on", "var_Q", "var_Q", "means"])
    assert [type(probe) for probe in scalars] == [OrderParameters, BlockVariances] and artifacts == []
    assert scalars[0].names == ["M_on", "Q_on", "G_on", "M_off", "Q_T", "G_T", "Q_N", "G_N"]
    assert scalars[1].names == ["var_Q"]
    assert OrderParameters().names == list(ORDER_PARAMS)
    assert order_parameter_probes([]) == ([], [])
    assert isinstance(order_parameter_probes(["profile"])[1][0], MProfile)
    for wrong in (["M_typo"], ["var_X"]):
        with pytest.raises(ValueError):
            order_parameter_probes(wrong)
    with pytest.raises(ValueError):
        BlockVariances(["M_on"])
