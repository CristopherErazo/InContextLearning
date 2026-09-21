"""Checks on the built-in attention masks.

The first test is the refactor's safety net: with `mask1 = mask2 = "causal"` the
model must reproduce exactly what it produced when the strictly lower-triangular
mask was carried in the batch and passed in by hand.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from icl import MinimalTransformer, make_mask


@dataclass
class Args:
    vocab_size: int = 32
    d_model: int = 16
    seq_len: int = 12
    lin_attn: bool = True
    beta: float = 0.25
    sigma_0: float = 1.0
    pred_mode: str = "next"
    dropout: float = 0.0
    mask1: str = "causal"
    mask2: str = "causal"


def build(**kw):
    torch.manual_seed(0)
    model = MinimalTransformer(Args(**kw))
    model.initialize_model()
    model.eval()
    x = torch.randint(0, model.vocab_size, (4, model.seq_len))
    return model, x


# ---- the mask helper ---------------------------------------------------------

def test_make_mask_kinds():
    causal, prev = make_mask("causal", 5), make_mask("prev", 5)
    assert torch.equal(causal, torch.tril(torch.ones(5, 5, dtype=torch.bool), diagonal=-1))
    assert torch.equal(prev, torch.diag(torch.ones(4, dtype=torch.bool), -1))
    assert not causal[0].any() and not prev[0].any()  # position 0 attends to nothing
    with torch.no_grad():
        try:
            make_mask("banana", 5)
        except ValueError:
            return
    raise AssertionError("an unknown mask kind must raise")


# ---- equivalence with the old externally-supplied mask -----------------------

def test_builtin_mask_reproduces_the_supplied_one():
    model, x = build()
    external = torch.tril(torch.ones(model.seq_len, model.seq_len, dtype=torch.bool), diagonal=-1)[None]
    with torch.no_grad():
        assert torch.equal(model(x), model(x, external))


def test_builtin_mask_reproduces_the_supplied_one_in_last_mode():
    model, x = build(pred_mode="last")
    external = torch.tril(torch.ones(model.seq_len, model.seq_len, dtype=torch.bool), diagonal=-1)[None]
    with torch.no_grad():
        out = model(x)
        assert out.shape == (x.size(0), 1, model.vocab_size)
        assert torch.equal(out, model(x, external))


def test_the_buffer_is_not_in_the_state_dict():
    """Snapshots written before the buffer existed must still load."""
    model, _ = build()
    assert not [k for k in model.state_dict() if k.endswith("attn1.mask")]
    assert model.attn1.mask.shape == (model.seq_len, model.seq_len)


# ---- per-layer masks ---------------------------------------------------------

def test_prev_restricts_layer_one_to_the_subdiagonal():
    model, x = build(mask1="prev")
    with torch.no_grad():
        out = model.full_output(x)
    A1, A2 = out["A1"], out["A2"]
    off_band = ~make_mask("prev", model.seq_len)
    assert (A1[:, off_band] == 0).all(), "layer 1 must only see the previous token"
    assert (A1[:, make_mask("prev", model.seq_len)] != 0).any()
    # layer 2 keeps its own, wider mask. Its key at position 0 is X1[:, 0], which is
    # zero under 'prev' (nothing to attend to), so probe a later key instead.
    assert (A2[:, ~make_mask("causal", model.seq_len)] == 0).all()
    assert (A2[:, 3, 1] != 0).any()


def test_shorter_input_than_seq_len():
    model, _ = build()
    x = torch.randint(0, model.vocab_size, (3, 5))
    with torch.no_grad():
        assert model(x).shape == (3, 5, model.vocab_size)


# ---- the softmax path --------------------------------------------------------

def test_softmax_path_has_no_nan():
    for kind in ("causal", "prev"):
        model, x = build(lin_attn=False, mask1=kind, mask2=kind)
        with torch.no_grad():
            out = model.full_output(x)
        for name in ("A1", "A2", "logits"):
            assert torch.isfinite(out[name]).all(), f"{name} is not finite with mask {kind!r}"
        assert (out["A1"][:, 0, :] == 0).all(), "a fully masked row must attend to nothing"
