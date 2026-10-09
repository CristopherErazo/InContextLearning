"""The logit noise of the variance ansatz (icl.theory.noise): no variance means no
noise; the formulas against the exact covariance given the sequence (eq. cov_exact
of archive/scratch/2026-10-05-0036_variance-profile-ansatz-explicit.tex) on real
sequences, at two points of the ansatz; that exact covariance against the model on
noisy ansatz matrices; the initialisation floor; the draws of `sample_logits`."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from icl import (VARIANCES, QueryTable, ReducedTransformer, ansatz_logits, ansatz_matrices, at_level, block_variance,
                 canonical_permutation, generate_icl_batch, logit_table, measure_variables, noise_variances,
                 non_trigger_loss, sample_logits, sample_variables, trigger_queries)


@dataclass
class Args:
    vocab_size: int = 128
    seq_len: int = 128
    lin_attn: bool = True
    beta: float = 0.25
    pred_mode: str = "next"
    dropout: float = 0.0
    mask1: str = "causal"
    mask2: str = "causal"


V, L, K, BETA = 128, 128, 25, 0.25
MUS = (64, 128)
PREFACTOR = (BETA / L) ** 2
POINTS = {      # (means, standard deviations of the blocks)
    "spread": ({"M_on": 1.0, "M_off": 0.1, "Q_on": 3.0, "Q_T": 0.5, "G_on": 2.0, "G_T": -0.7},
               {"M_on": 0.5, "M_off": 0.7, "Q_on": 0.8, "Q_T": 0.6, "Q_N": 0.4, "G_on": 0.5, "G_T": 0.6, "G_N": 0.3}),
    "trained": ({"M_on": 2.0, "M_off": 0.05, "Q_on": 8.0, "Q_T": 0.3, "G_on": 5.0, "G_T": -1.0},
                {"M_on": 0.3, "M_off": 0.3, "Q_on": 0.5, "Q_T": 0.3, "Q_N": 0.3, "G_on": 0.5, "G_T": 0.3, "G_N": 0.3}),
}


def order_params(point: str) -> dict:
    """The means with a decreasing previous-token profile and the eight block variances."""
    means, sds = POINTS[point]
    profile = means["M_on"] * (1.6 - 1.2 * torch.arange(2, L + 1, dtype=torch.float64) / L)
    return {**means, "M_profile": profile, **{"var_" + name: sd ** 2 for name, sd in sds.items()}}


def entry_variances(order_params: dict) -> dict[str, torch.Tensor]:
    """The variance of every entry of M, Q, G under the variance ansatz."""
    variances = {"M": torch.zeros(L, L, dtype=torch.float64), "Q": torch.zeros(V, V, dtype=torch.float64),
                 "G": torch.zeros(V, V, dtype=torch.float64)}
    for name, (matrix, support) in VARIANCES.items():
        variances[matrix][support(L, V, K)] = float(block_variance(order_params, name))
    return variances


def exact_noise(rows: torch.Tensor, mu: int, means: dict, variances: dict):
    """The exact covariance of xi given the sequences `rows` (B, >= mu) at the query mu
    (eqs. cov_exact, Calpha): the variance of every logit (B, V) in token order, the
    covariance C^alpha of the key amplitudes (B, n, n), the mean readout of every key by
    every logit (B, n, V) and the column energies (B, V)."""
    p = mu - 1                                                         # code position of the query
    query, sources, keys = rows[:, p], rows[:, :p - 1], rows[:, 1:p]   # key i <-> code i + 1, source j <-> code j
    M, var_M = means["M"][1:p, :p - 1], variances["M"][1:p, :p - 1]
    Q_row, var_Q_row = means["Q"][query], variances["Q"][query]
    Q_sources, var_Q_sources = Q_row.gather(1, sources), var_Q_row.gather(1, sources)
    amplitude = Q_sources @ M.T                                        # A_bar_nu
    incoherent = (Q_sources ** 2 + var_Q_sources) @ var_M.T            # d_nu
    by_token = torch.einsum("ij,bjc->bic", M, F.one_hot(sources, V).double())     # S_nu(c)
    C = torch.einsum("bic,bc,bkc->bik", by_token, var_Q_row, by_token) + torch.diag_embed(incoherent)
    readout = means["G"].T[keys]
    holds = F.one_hot(keys, V).double()
    energy = torch.einsum("bi,bic->bc", amplitude, holds) ** 2 + torch.einsum("bic,bik,bkc->bc", holds, C, holds)
    variance = PREFACTOR * (torch.einsum("bit,bik,bkt->bt", readout, C, readout) + energy @ variances["G"].T)
    return variance, C, readout, energy


@pytest.mark.parametrize("point", list(POINTS))
def test_formulas_match_the_exact_covariance(point):
    op = order_params(point)
    means, variances = ansatz_matrices(op, L, V, K), entry_variances(op)
    batch = generate_icl_batch(2048, V, L, K, generator=torch.Generator().manual_seed(0))
    inputs = batch["sequence"][:, :-1]
    for mu in MUS:
        sequence_index, _ = trigger_queries(inputs, K, [mu])
        variables = measure_variables(batch, [mu])                     # rows in the same order
        formulas = noise_variances(variables, op, BETA, L)
        variance, C, readout, _ = exact_noise(inputs[sequence_index], mu, means, variances)
        permutation = canonical_permutation(inputs[sequence_index, mu - 1], batch["output_set"][sequence_index], K, V)
        canonical = variance.gather(1, permutation)
        readout = readout.gather(2, permutation[:, None, :].expand_as(readout))
        common = PREFACTOR * torch.einsum("bi,bik,bk->b", readout[:, :, 0], C, readout[:, :, 0])
        exact = {"target": canonical[:, -1], "trigger_common": common,
                 "trigger_individual": canonical[:, :K].mean(1) - common, "non_target": canonical[:, K:V - 1],
                 "target_trigger": PREFACTOR * torch.einsum("bi,bik,bk->b", readout[:, :, -1], C, readout[:, :, 0])}
        for name, value in exact.items():
            # leading order: R1-R3 replace sums over the sequence by their means, a few % row by row
            assert formulas[name].mean() / value.mean() == pytest.approx(1, abs=0.03), (point, mu, name)
        in_context = variables["ell"] >= 1                             # where the witness corrections matter
        ratio = formulas["target_trigger"][in_context].mean() / exact["target_trigger"][in_context].mean()
        assert ratio == pytest.approx(1, abs=0.04), (point, mu)

        # non-trigger queries: half the variance of the logit vector about its mean (eq. LN_exact2)
        rows = inputs[inputs[:, mu - 1] >= K][:1024]
        variance, C, readout, energy = exact_noise(rows, mu, means, variances)
        mean_readout = readout.mean(2)
        spread = variance.mean(1) - PREFACTOR * (torch.einsum("bi,bik,bk->b", mean_readout, C, mean_readout)
                                                 + energy @ variances["G"].sum(0) / V ** 2)
        formula = non_trigger_loss(mu, op, BETA, L, V, K) - math.log(V)
        assert formula.item() / (spread.mean().item() / 2) == pytest.approx(1, abs=0.02), (point, mu)


def test_the_exact_covariance_is_the_models():
    """The logits of the model on draws of the variance ansatz scatter around the mean
    logits with the exact covariance, annealed over the draws: one network is off by
    20-40% on the target (only K rows of Q are read at trigger queries), the mean of 12
    by about 7%."""
    op = order_params("spread")
    means, variances = ansatz_matrices(op, L, V, K), entry_variances(op)
    batch = generate_icl_batch(512, V, L, K, generator=torch.Generator().manual_seed(1))
    mu = L
    variables = measure_variables(batch, [mu])
    sequence_index, _ = trigger_queries(batch["sequence"][:, :-1], K, [mu])
    variance, *_ = exact_noise(batch["sequence"][sequence_index, :-1], mu, means, variances)
    permutation = canonical_permutation(batch["sequence"][sequence_index, mu - 1], batch["output_set"][sequence_index],
                                        K, V)
    exact = variance.gather(1, permutation)
    squares = []
    for seed in range(12):
        model = ReducedTransformer.from_matrices(ansatz_matrices(op, L, V, K, noise=torch.Generator().manual_seed(seed)),
                                                 Args())
        xi = logit_table(model, batch, [mu])["logits"] - ansatz_logits(variables, op, BETA, L)
        squares.append(xi ** 2)
    measured = torch.stack(squares).mean(0)
    for block in (slice(0, K), slice(K, V - 1), slice(V - 1, V)):
        assert measured[:, block].mean() / exact[:, block].mean() == pytest.approx(1, abs=0.2), block


def test_no_variance_no_noise():
    means = {**POINTS["spread"][0], "M_profile": order_params("spread")["M_profile"]}
    batch = generate_icl_batch(64, V, L, K, generator=torch.Generator().manual_seed(2))
    for variables in (measure_variables(batch, MUS), sample_variables(L, 2, V, K, num_samples=64)):
        for op in (means, at_level({**means, "var_M": 1.0}, "S5")):
            for value in noise_variances(variables, op, BETA, L).values():
                assert torch.equal(value, torch.zeros_like(value))
            assert torch.equal(sample_logits(variables, op, BETA, L), ansatz_logits(variables, op, BETA, L))
    assert torch.equal(non_trigger_loss(torch.arange(1, L + 1), means, BETA, L, V, K),
                       torch.full((L,), math.log(V), dtype=torch.float64))
    # a variance at 0 has a finite gradient (no 0/0 in the standard deviations)
    variance = torch.zeros((), dtype=torch.float64, requires_grad=True)
    sample_logits(variables, {**means, "var_Q_T": variance}, BETA, L).sum().backward()
    assert torch.isfinite(variance.grad)


def test_pooled_variances_fill_the_blocks():
    variables = sample_variables(100, 2, V, K, num_samples=32, generator=torch.Generator().manual_seed(3))
    means = POINTS["spread"][0]
    pooled = {"var_M": 0.3, "var_Q": 0.5, "var_G": 0.2}
    by_block = {name: pooled["var_" + matrix] for name, (matrix, _) in VARIANCES.items()}
    one, other = (noise_variances(variables, {**means, **values}, BETA, L) for values in (pooled, by_block))
    for name in one:
        assert torch.equal(one[name], other[name]), name


def test_initialisation_floor():
    """All means 0 and one variance per matrix: every entry has V_0 = (beta/L)^2
    s_G^2 s_Q^2 s_M^2 (mu-1)(mu-2)/2 and nothing is shared (eq. V0)."""
    op = {"var_M": 0.5, "var_Q": 2.0, "var_G": 3.0}
    for mu in (3, 50, L):
        variables = sample_variables(mu, 1, V, K, num_samples=8, generator=torch.Generator().manual_seed(mu))
        floor = PREFACTOR * 3.0 * 2.0 * 0.5 * (mu - 1) * (mu - 2) / 2
        variances = noise_variances(variables, op, BETA, L)
        for name in ("target", "trigger_individual", "non_target"):
            torch.testing.assert_close(variances[name], torch.full_like(variances[name], floor), rtol=1e-12, atol=0)
        for name in ("trigger_common", "target_trigger"):
            assert torch.equal(variances[name], torch.zeros_like(variances[name]))
        expected = math.log(V) + 0.5 * (1 - 1 / V) * floor
        assert non_trigger_loss(mu, op, BETA, L, V, K).item() == pytest.approx(expected, rel=1e-14)


def test_sample_logits_draws_the_covariance():
    """With the unit vectors as normals the draws are the columns of the Cholesky factor,
    so their outer products sum to the covariance of the note exactly."""
    op = order_params("spread")
    row = sample_variables(L, 2, V, K, num_samples=1, generator=torch.Generator().manual_seed(4))
    rows = QueryTable.concatenate([row] * (V + 1))
    xi = sample_logits(rows, op, BETA, L, normals=torch.eye(V + 1, dtype=torch.float64)) - ansatz_logits(rows, op,
                                                                                                         BETA, L)
    variances = {name: value[0] for name, value in noise_variances(row, op, BETA, L).items()}
    expected = torch.diag(torch.cat([variances["trigger_individual"].expand(K), variances["non_target"],
                                     variances["target"][None]]))
    expected[:K, :K] += variances["trigger_common"]
    expected[:K, -1] = expected[-1, :K] = variances["target_trigger"]
    torch.testing.assert_close(xi.T @ xi, expected, rtol=1e-10, atol=1e-16)


def test_triggered_output_pairs():
    """Y_bar: measured, the pairs of occurrences of an output after its trigger (none for
    the tokens that are no output); sampled and mean, their mean (p_T mu)^2."""
    batch = generate_icl_batch(1024, V, L, K, generator=torch.Generator().manual_seed(5))
    measured = measure_variables(batch, [L])
    pairs, counts = measured["Y_bar"], measured["U_bar"]
    assert (pairs <= counts * (counts - 1)).all() and (pairs[:, K - 1:] == 0).all()
    expected = (L / (V + K)) ** 2
    assert pairs[:, :K - 1].mean().item() == pytest.approx(expected, rel=0.1)
    sampled = sample_variables(L, 1, V, K, num_samples=4096, generator=torch.Generator().manual_seed(6))
    assert sampled["Y_bar"][:, :K - 1].mean().item() == pytest.approx(expected, rel=0.05)
