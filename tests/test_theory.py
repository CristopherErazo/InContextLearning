"""The effective model against the model itself.

The central check (test_ansatz_logits_are_exact): put the matrices exactly in
the ansatz (`ansatz_matrices`), run the real forward pass, and compare with
`ansatz_logits` evaluated on the variables measured on the same sequences.
There is no background term, so the two must agree to rounding; any difference
is a bug in the logit formula, in the variable definitions, in the canonical
layout or in the mu indexing. Keep it passing when the ansatz is extended.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from icl import (ORDER_PARAMS, Evaluator, OrderParameters, QueryTable, ReducedTransformer, TriggerLogitTable,
                 ansatz_logits, ansatz_matrices, canonical_permutation, cluster_cross_entropy,
                 generate_icl_batch, logit_blocks, logit_table, measure_order_params, measure_variables,
                 preprocess_batch, support_sizes)


@dataclass
class Args:
    vocab_size: int = 24
    seq_len: int = 20
    d_model: int = 48
    lin_attn: bool = True
    beta: float = 0.25
    pred_mode: str = "next"
    dropout: float = 0.0
    mask1: str = "causal"
    mask2: str = "causal"


V, L, K = 24, 20, 4


def random_order_params(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {name: torch.randn(1, generator=generator, dtype=torch.float64).item() for name in ORDER_PARAMS}


def ansatz_model(order_params, **overrides):
    args = Args(**overrides)
    return ReducedTransformer.from_matrices(ansatz_matrices(order_params, L, V, K), args), args


# ---- the layout ---------------------------------------------------------------

def test_canonical_permutation():
    torch.manual_seed(0)
    num_rows = 50
    output_sets = torch.stack([torch.randperm(V - K)[:K] + K for _ in range(num_rows)])
    query_tokens = torch.randint(0, K, (num_rows,))
    permutation = canonical_permutation(query_tokens, output_sets, K, V)
    blocks = logit_blocks(K, V)
    assert (permutation.sort(dim=1).values == torch.arange(V)).all()     # a permutation
    for row in range(num_rows):
        query, outputs = query_tokens[row].item(), output_sets[row]
        assert permutation[row, blocks["triggers"]].tolist() == list(range(K))
        assert permutation[row, blocks["other_outputs"]].tolist() == [outputs[c].item() for c in range(K) if c != query]
        assert permutation[row, blocks["rest"]].tolist() == sorted(set(range(K, V)) - set(outputs.tolist()))
        assert permutation[row, -1].item() == outputs[query].item()


# ---- the registry -------------------------------------------------------------

def test_order_params_round_trip_and_sizes():
    order_params = random_order_params()
    measured = measure_order_params(ansatz_matrices(order_params, L, V, K), K)
    assert measured.keys() == order_params.keys()
    for name in order_params:
        assert measured[name] == pytest.approx(order_params[name], rel=1e-12)
    assert support_sizes(L, V, K) == {"M_on": L - 1, "M_off": (L - 1) * (L - 2) // 2, "Q_on": K,
                                      "Q_T": K * (V - 1), "G_on": V - K, "G_T": K * V}


# ---- the exactness test -------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("pred_mode", ["next", "last"])
def test_ansatz_logits_are_exact(seed, pred_mode):
    order_params = random_order_params(seed)
    model, args = ansatz_model(order_params, pred_mode=pred_mode)
    batch = generate_icl_batch(256, V, L, K, generator=torch.Generator().manual_seed(seed))
    mus = None if pred_mode == "next" else L
    model_logits = logit_table(model, batch, mus=mus, chunk=37)
    variables = measure_variables(batch, mus=mus, chunk=101)
    for column in ("mu", "ell"):
        assert torch.equal(model_logits[column], variables[column].cpu())
    assert (model_logits["ell"] == 0).any() and (model_logits["ell"] >= 2).any()   # every cluster type
    if pred_mode == "next":
        assert set(model_logits["mu"].tolist()) >= {1, 2, 3, L}
    expected = ansatz_logits(variables, order_params, args.beta, args.seq_len)
    torch.testing.assert_close(model_logits["logits"], expected, rtol=1e-12, atol=1e-12)


def test_non_trigger_queries_have_zero_logits():
    model, _ = ansatz_model(random_order_params())
    inputs = generate_icl_batch(64, V, L, K)["sequence"][:, :-1]
    with torch.no_grad():
        logits = model(inputs)
    assert (logits[inputs >= K] == 0).all()


def test_measured_variables_by_hand():
    """One sequence written out: query a=0 with phi(0)=5, V=24, K=4."""
    sequence = torch.tensor([[0, 5, 9, 5, 0, 5, 7, 9, 0, 5]])   # inputs at nu = 1..9, query a=0 at mu=9
    output_set = torch.tensor([[5, 6, 7, 8]])
    batch = {"sequence": sequence, "output_set": output_set, "K": K, "V": V}
    variables = measure_variables(batch, mus=9)
    assert variables.num_rows == 1 and variables["mu"].item() == 9 and variables["ell"].item() == 2
    # earlier a at nu = 1, 5; phi(a)=5 at the keys nu = 2, 4, 6, the one at nu = 4 is free
    assert variables["N"].item() == 2 and variables["F"].item() == 1
    assert variables["R"].item() == (9 - 1 - 2) + (9 - 5 - 2)
    assert variables["W"].item() == (2 - 2) + (4 - 2) + (6 - 2)
    assert variables["P"].item() == 0 + 1 + 1                  # a's at nu <= key - 2 for the keys 2, 4, 6
    non_target = logit_blocks(K, V)["non_target"]
    tokens_in_order = canonical_permutation(torch.tensor([0]), output_set, K, V)[0, non_target]
    column_of = {token.item(): column for column, token in enumerate(tokens_in_order)}
    # token 9 at nu = 3, 8; token 7 (= phi(2), an "other_outputs" token) at nu = 7
    assert variables["U_bar"][0, column_of[9]].item() == 2
    assert variables["W_bar"][0, column_of[9]].item() == 1 + 6
    assert variables["P_bar"][0, column_of[9]].item() == 1 + 2
    assert variables["U_bar"][0, column_of[7]].item() == 1 and variables["P_bar"][0, column_of[7]].item() == 2


# ---- the table and the cross-entropy ------------------------------------------

def test_query_table():
    table = QueryTable({"mu": torch.tensor([1, 2, 2, 3]), "ell": torch.tensor([0, 1, 1, 0]),
                        "value": torch.arange(4.0), "K": 4})
    assert table.select(mu=2)["value"].tolist() == [1.0, 2.0] and table.select(mu=2)["K"] == 4
    assert table.select(mu=[1, 3], ell=0)["value"].tolist() == [0.0, 3.0]
    assert [key for key, _ in table.groups("mu", "ell")] == [(1, 0), (2, 1), (3, 0)]
    reloaded = QueryTable.from_numpy(table.to_numpy())
    assert torch.equal(reloaded["value"], table["value"]) and reloaded["K"] == 4
    assert QueryTable.concatenate([table, table]).num_rows == 8


def test_cluster_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(30, V, dtype=torch.float64, requires_grad=True)
    table = QueryTable({"logits": logits, "mu": torch.randint(1, 4, (30,)), "ell": torch.randint(0, 2, (30,))})
    clusters = cluster_cross_entropy(table)
    for cluster in range(clusters.num_rows):
        in_cluster = (table["mu"] == clusters["mu"][cluster]) & (table["ell"] == clusters["ell"][cluster])
        expected = (torch.logsumexp(logits[in_cluster], 1) - logits[in_cluster, -1]).mean()
        assert clusters["count"][cluster].item() == in_cluster.sum().item()
        torch.testing.assert_close(clusters["cross_entropy"][cluster], expected)
    clusters["cross_entropy"].sum().backward()
    assert logits.grad is not None


# ---- the probes and the test batch --------------------------------------------

def test_probes():
    order_params = random_order_params(3)
    model, _ = ansatz_model(order_params)
    test_batch, stats = preprocess_batch(generate_icl_batch(64, V, L, K))
    assert stats.n_sequences == 64                       # the test batch is not filtered
    evaluator = Evaluator(scalars=[OrderParameters()], artifacts=[TriggerLogitTable([0.5, 1.0])], chunk=9)
    metrics = evaluator.scalars(model, test_batch, step=0)
    assert metrics.keys() == order_params.keys()
    for name in order_params:
        assert metrics[name] == pytest.approx(order_params[name], rel=1e-12)
    saved = QueryTable.from_numpy(evaluator.artifacts(model, test_batch, step=0)[("table", "logits")][0])
    expected = logit_table(model, test_batch, mus=[math.ceil(0.5 * L), L])
    for column in ("logits", "mu", "ell", "sequence_index"):
        assert torch.equal(saved[column], expected[column])
    assert (saved["ell"] == 0).any()                     # trigger queries with no earlier occurrence are kept
