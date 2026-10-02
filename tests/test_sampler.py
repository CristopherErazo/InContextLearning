"""The sampler of the sequence variables against the moments of the scratch
file (Table 2 and the covariances after it). With Poisson counts the sampler
draws exactly the law those moments were derived from, so every mean, variance
and covariance must match within Monte Carlo error.
"""
from __future__ import annotations

import pytest
import torch

from icl import (ORDER_PARAMS, ansatz_logits, logit_blocks, mean_variables, sample_variables)

V, K = 64, 12
NUM_SAMPLES = 40_000
Z_MAX = 4.5


def z_score(estimate, expected, standard_error):
    return abs(estimate - expected) / standard_error


def check_mean(name, values, expected):
    if values.std() == 0:                      # deterministic: must be exact
        assert values[0].item() == pytest.approx(expected), f"{name}: {values[0]:.4g} vs {expected:.4g}"
        return
    se = values.std() / len(values) ** 0.5
    assert z_score(values.mean(), expected, se) < Z_MAX, f"mean of {name}: {values.mean():.4g} vs {expected:.4g}"


def check_variance(name, values, expected):
    centered = values - values.mean()
    if expected == 0:
        assert centered.abs().max() < 1e-9, f"{name} should be deterministic"
        return
    se = ((centered ** 2).var() / len(values)) ** 0.5
    assert z_score(centered.pow(2).mean(), expected, se) < Z_MAX, \
        f"variance of {name}: {centered.pow(2).mean():.4g} vs {expected:.4g}"


def check_covariance(name, first, second, expected):
    product = (first - first.mean()) * (second - second.mean())
    if product.std() == 0:                     # one of the two is deterministic
        assert expected == pytest.approx(0.0), f"cov {name} should vanish"
        return
    se = product.std() / len(product) ** 0.5
    assert z_score(product.mean(), expected, se) < Z_MAX, f"cov {name}: {product.mean():.4g} vs {expected:.4g}"


@pytest.mark.parametrize("mu,ell", [(200, 0), (200, 1), (150, 3)])
def test_poisson_sampler_matches_the_moments(mu, ell):
    generator = torch.Generator().manual_seed(mu + ell)
    variables = sample_variables(mu, ell, V, K, num_samples=NUM_SAMPLES, generator=generator, chunk=7000)
    assert variables.num_rows == NUM_SAMPLES and variables["U_bar"].shape == (NUM_SAMPLES, V - K - 1)
    m = mu / (V + K)
    N, F, R, W, P = (variables[name] for name in ("N", "F", "R", "W", "P"))
    expected = {   # name: (values, mean, variance)
        "N": (N, ell, 0.0),
        "F": (F, m, m),
        "R": (R, mu * ell / 2, mu ** 2 * ell / 12),
        "W": (W, mu * (ell + m) / 2, mu ** 2 * (ell / 12 + m / 3)),
        "P": (P, ell * (ell - 1) / 2 + m * ell / 2, m * ell * (2 * ell + 1) / 6 + m ** 2 * ell / 12),
    }
    # one token of each kind: an output of another trigger (rate 2m) and a rest token (rate m)
    for column, rate in ((0, 2 * m), (K - 1, m)):
        U, W_bar, P_bar = (variables[name][:, column] for name in ("U_bar", "W_bar", "P_bar"))
        expected[f"U_bar[{column}]"] = (U, rate, rate)
        expected[f"W_bar[{column}]"] = (W_bar, mu * rate / 2, mu ** 2 * rate / 3)
        expected[f"P_bar[{column}]"] = (P_bar, rate * ell / 2, rate * ell * (2 * ell + 1) / 6 + rate ** 2 * ell / 12)
        check_covariance(f"U_bar, P_bar [{column}]", U, P_bar, rate * ell / 2)
        check_covariance(f"U_bar, W_bar [{column}]", U, W_bar, mu * rate / 2)
        check_covariance(f"W_bar, P_bar [{column}]", W_bar, P_bar, mu * rate * ell / 3)
        if ell:
            check_covariance(f"R, P_bar [{column}]", R, P_bar, rate * mu * ell / 12)
    for name, (values, mean, variance) in expected.items():
        check_mean(name, values, mean)
        check_variance(name, values, variance)
    check_covariance("F, W", F, W, mu * m / 2)
    check_covariance("F, P", F, P, m * ell / 2)
    if ell:
        check_covariance("R, W", R, W, -mu ** 2 * ell / 12)
        check_covariance("R, P", R, P, m * mu * ell / 12)
        check_covariance("W, P", W, P, mu * m * ell / 4)


def test_mean_variables_are_the_sample_means():
    mean = mean_variables([200, 150], [1, 3], V, K)
    assert mean.num_rows == 2 and mean["U_bar"].shape == (2, V - K - 1)
    for row, (mu, ell) in enumerate([(200, 1), (150, 3)]):
        variables = sample_variables(mu, ell, V, K, num_samples=NUM_SAMPLES, generator=torch.Generator().manual_seed(row))
        for name in ("F", "R", "W", "P"):
            check_mean(name, variables[name], mean[name][row].item())
        for name in ("U_bar", "W_bar", "P_bar"):
            check_mean(name, variables[name][:, K], mean[name][row, K].item())


def test_multinomial_counts_share_the_available_keys():
    mu, ell = 200, 2
    variables = sample_variables(mu, ell, V, K, num_samples=NUM_SAMPLES, counts="multinomial",
                                 generator=torch.Generator().manual_seed(0))
    available = mu - 2 - 2 * ell
    # stationary weights in units of p_T: free target 1, other outputs 2, rest 1, other triggers K - 1
    weights = torch.tensor([1.0] + [2.0] * (K - 1) + [1.0] * (V - 2 * K) + [K - 1.0])
    probability = weights / weights.sum()
    total = variables["F"] + variables["U_bar"].sum(1)
    assert (total <= available).all()
    check_mean("U_bar[0]", variables["U_bar"][:, 0], available * probability[1].item())
    check_mean("F", variables["F"], available * probability[0].item())
    # the budget: the non-target total is binomial, narrower than a sum of independent Poissons
    share = probability[1:-1].sum().item()
    check_variance("sum U_bar", variables["U_bar"].sum(1), available * share * (1 - share))


def test_broadcasting_reproducibility_and_errors():
    variables = sample_variables(torch.tensor([10, 50, 90]), 2, V, K, generator=torch.Generator().manual_seed(1))
    assert variables["mu"].tolist() == [10, 50, 90] and variables["ell"].tolist() == [2, 2, 2]
    again = sample_variables(torch.tensor([10, 50, 90]), 2, V, K, generator=torch.Generator().manual_seed(1))
    assert torch.equal(variables["W_bar"], again["W_bar"])
    zero = sample_variables(30, 0, V, K, num_samples=5)
    assert (zero["R"] == 0).all() and (zero["P_bar"] == 0).all() and (zero["P"] == 0).all()
    with pytest.raises(ValueError):
        sample_variables(10, 1, V, K, num_samples=5, counts="binomial")
    with pytest.raises(ValueError):
        sample_variables(0, 1, V, K, num_samples=5)
    with pytest.raises(ValueError):
        sample_variables(torch.tensor([10, 20]), torch.tensor([1, 2, 3]), V, K)


def test_ansatz_logits_on_sampled_and_mean_variables():
    """The logits are linear in the variables: the mean of the sampled logits is
    the logit of the mean variables."""
    order_params = {name: value for name, value in zip(ORDER_PARAMS, (1.3, 0.4, 2.0, -0.3, 1.1, -0.5))}
    mu, ell, beta, L = 120, 2, 0.25, 128
    variables = sample_variables(mu, ell, V, K, num_samples=NUM_SAMPLES, generator=torch.Generator().manual_seed(2))
    sampled = ansatz_logits(variables, order_params, beta, L)
    at_mean = ansatz_logits(mean_variables(mu, ell, V, K), order_params, beta, L)[0]
    assert sampled.shape == (NUM_SAMPLES, V)
    blocks = logit_blocks(K, V)
    for block in ("triggers", "other_outputs", "rest", "target"):
        column = blocks[block].start
        check_mean(block, sampled[:, column], at_mean[column].item())
