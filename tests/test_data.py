"""Light checks on the vectorised batch generator.

Two kinds of assertion: exact ones on the structure the task guarantees (the
trigger rule, distinct output sets, `counts`), and loose statistical ones (five
standard errors) on the laws that only hold in distribution -- including
agreement with the pre-refactor loop implementation in `reference_data.py`.
"""
from __future__ import annotations

import math

import torch

from icl.data import generate_icl_batch, occurrence_counts
from reference_data import generate_icl_batch_loop

V, L, K = 64, 48, 12
B = 4000


def gen(**kw):
    g = torch.Generator().manual_seed(kw.pop("seed", 0))
    return generate_icl_batch(kw.pop("B", B), V, L, K, generator=g, **kw)


def five_sigma(p: float, n: int) -> float:
    """Tolerance on an empirical frequency of a probability-p event over n draws."""
    return 5 * math.sqrt(max(p * (1 - p), 1e-12) / n)


# ---- structure ---------------------------------------------------------------

def test_keys_and_shapes():
    batch = gen()
    assert set(batch) == {"sequence", "K", "output_set", "counts", "is_trigg"}
    assert batch["sequence"].shape == (B, L + 1)
    assert batch["counts"].shape == batch["is_trigg"].shape == (B, L)
    assert batch["output_set"].shape == (B, K)
    assert batch["K"] == K
    assert "mask" not in batch, "the mask belongs to the model, not to the batch"


def test_stats_false_skips_the_bookkeeping():
    batch = gen(stats=False)
    assert set(batch) == {"sequence", "K", "output_set"}
    # and gives the same sequences as stats=True for the same stream
    assert torch.equal(batch["sequence"], gen()["sequence"])


def test_output_sets_are_distinct_non_triggers():
    out = gen()["output_set"]
    assert (out >= K).all() and (out < V).all()
    assert all(len(row.unique()) == K for row in out)


def test_trigger_rule_is_exact():
    batch = gen()
    seq, is_trigg = batch["sequence"], batch["is_trigg"]
    b_idx, t_idx = torch.where(is_trigg)
    expected = batch["output_set"][b_idx, seq[b_idx, t_idx]]
    assert torch.equal(seq[b_idx, t_idx + 1], expected)


def test_is_trigg_and_no_two_in_a_row():
    batch = gen()
    seq, is_trigg = batch["sequence"], batch["is_trigg"]
    assert torch.equal(is_trigg, seq[:, :L] < K), "the triggers are exactly the tokens 0..K-1"
    assert not (is_trigg[:, :-1] & is_trigg[:, 1:]).any(), "an output is never a trigger"


def test_counts_match_a_naive_count():
    x = gen(B=32)["sequence"]
    naive = torch.zeros_like(x)
    for b in range(x.size(0)):
        seen: dict[int, int] = {}
        for t in range(x.size(1)):
            tok = int(x[b, t])
            seen[tok] = seen.get(tok, 0) + 1
            naive[b, t] = seen[tok]
    assert torch.equal(occurrence_counts(x), naive)


def test_generator_is_reproducible():
    assert torch.equal(gen(seed=7)["sequence"], gen(seed=7)["sequence"])
    assert not torch.equal(gen(seed=7)["sequence"], gen(seed=8)["sequence"])


# ---- distributions -----------------------------------------------------------

def test_position_zero_is_stationary():
    """pi(output token) = 2/(V+K), so the first token is an output w.p. 2K/(V+K)."""
    batch = gen(B=20000)
    first = batch["sequence"][:, 0]
    is_out = (first.unsqueeze(1) == batch["output_set"]).any(dim=1).float().mean().item()
    p = 2 * K / (V + K)
    assert abs(is_out - p) < five_sigma(p, 20000)


def test_non_trigger_successors_are_uniform():
    batch = gen(B=20000)
    seq, is_trigg = batch["sequence"], batch["is_trigg"]
    nxt = seq[:, 1:][~is_trigg]
    freq = torch.bincount(nxt, minlength=V).float() / nxt.numel()
    assert (freq - 1 / V).abs().max().item() < five_sigma(1 / V, nxt.numel())


def test_agrees_with_the_loop_implementation():
    torch.manual_seed(0)
    ref = generate_icl_batch_loop(B, V, L, K)
    new = gen(B=B)

    for name, value in (("trigger rate", lambda d: (d["is_trigg"].bool()).float()),
                        ("induction possible", lambda d: (d["is_trigg"].bool() & (d["counts"] > 1)).float())):
        p_ref, p_new = value(ref).mean().item(), value(new).mean().item()
        n = B * L
        assert abs(p_ref - p_new) < 5 * math.sqrt(2 * max(p_ref * (1 - p_ref), 1e-12) / n), (
            f"{name}: reference {p_ref:.4f} vs new {p_new:.4f}")

    # token marginal over the whole batch
    f_ref = torch.bincount(ref["sequence"].flatten(), minlength=V).float() / ref["sequence"].numel()
    f_new = torch.bincount(new["sequence"].flatten(), minlength=V).float() / new["sequence"].numel()
    assert (f_ref - f_new).abs().max().item() < 5 * math.sqrt(2 * (1 / V) / (B * (L + 1)))
