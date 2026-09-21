"""Sampling for the trigger-retrieval task.

The sequence is a Markov chain on the vocabulary: a *trigger* token (always
`0..K-1`) is followed by its own output token, anything else is followed by a
uniform draw. Each sequence draws its own K distinct output tokens from
`[K, V-1]`, so **an output is never a trigger** and the chain can never emit two
triggers in a row. That invariant is what makes the generator loop-free.

Write `u_t` for an i.i.d. uniform proposal at position `t`, `b_t = 1{u_t < K}`
and `T_t = 1{x_t is a trigger}`. Because a trigger is always followed by a
non-trigger, `x_t = u_t` whenever `T_{t-1} = 0`, and therefore

    T_t = b_t and not T_{t-1}
    x_t = output_set[u_{t-1}]  if T_{t-1} else u_t

The first line says that inside a maximal run of `b = 1` the triggers sit at the
odd offsets, which is a `cummax` over "index of the last position with b = 0";
the second is a single gather. So the whole batch is drawn in a dozen
vectorised ops instead of an L-step Python loop.

The attention mask is *not* part of a batch: it is a constant of the model, not
of the data. Use `icl.make_mask` if you need one explicitly.
"""
from __future__ import annotations

import torch

Batch = dict[str, torch.Tensor]


def generate_icl_batch(num_samples: int,
                       V: int,
                       L: int,
                       K: int,
                       stats: bool = True,
                       device: str | torch.device = "cpu",
                       generator: torch.Generator | None = None,
                       ) -> Batch:
    """Batch generator for the trigger-retrieval task.

    Args:
        num_samples: batch size B.
        V: vocabulary size. L: sequence length. K: number of trigger tokens.
        stats: also return the `counts` / `is_trigg` bookkeeping. Training only
            reads `sequence`, so it can pass `stats=False` and skip that work;
            evaluation needs it to build the "induction possible" mask.
        device: where the batch is drawn (and stays).
        generator: optional `torch.Generator` for a reproducible stream
            independent of the global seed. Must live on `device`.

    Returns:
        dict with:
            sequence     : (B, L+1)   input is [:, :-1], target is [:, 1:]
            trigger_set  : (B, K)     always 0..K-1, the same for every sequence
            output_set   : (B, K)     trigger i of sequence b maps to [b, i]
            counts       : (B, L)     occurrences of the token at each position,
                                      counting that position (only if `stats`)
            is_trigg     : (B, L)     bool, token at this position is a trigger
                                      (only if `stats`)
    """
    B = num_samples

    if K > V - K:
        raise ValueError(f"need 2K <= V to draw K distinct outputs from [K, V-1], got V={V}, K={K}")

    rand_kw = {"device": device, "generator": generator}

    # ---- output tokens: K distinct draws from [K, V-1], one set per sequence ----
    # topk of uniform noise is a uniformly random K-subset in a random order,
    # and is markedly cheaper than multinomial(..., replacement=False).
    output_sets = torch.rand(B, V - K, **rand_kw).topk(K, dim=1).indices + K  # (B, K)

    # ---- uniform proposals, one per position ----
    u = torch.randint(0, V, (B, L + 1), **rand_kw)

    # Position 0 is drawn from the chain's stationary law, which puts weight
    # 2/(V+K) on an output token and 1/(V+K) on everything else. That is exactly
    # the mixture "uniform token w.p. V/(V+K), uniform output token w.p. K/(V+K)",
    # so it costs one Bernoulli and one gather rather than a (B, V) multinomial.
    take_output = torch.rand(B, **rand_kw) < K / (V + K)
    which = torch.randint(0, K, (B, 1), **rand_kw)
    u[:, 0] = torch.where(take_output, output_sets.gather(1, which).squeeze(1), u[:, 0])

    # ---- trigger flags: T_t = b_t and not T_{t-1}, by run parity ----
    pos = torch.arange(L + 1, device=device).expand(B, -1)          # (B, L+1)
    b = u < K
    last_zero = torch.cummax(torch.where(b, -torch.ones_like(pos), pos), dim=1).values
    run = pos - last_zero                # length of the run of b=1 ending here
    is_trigg = b & (run % 2 == 1)        # (B, L+1), == (sequence < K)

    # ---- the sequence itself ----
    # Where the previous position held a trigger, it was its own proposal, so the
    # output token is output_sets[u_{t-1}]; the clamp only keeps the gather legal
    # at the positions that torch.where discards.
    mapped = output_sets.gather(1, u[:, :L].clamp(max=K - 1))
    sequence = torch.cat([u[:, :1], torch.where(is_trigg[:, :L], mapped, u[:, 1:])], dim=1)

    batch: Batch = {
        "sequence": sequence,                                       # (B, L+1)
        "trigger_set": torch.arange(K, device=device).expand(B, -1),  # (B, K)
        "output_set": output_sets,                                  # (B, K)
    }
    if stats:
        batch["counts"] = occurrence_counts(sequence)[:, :L]
        batch["is_trigg"] = is_trigg[:, :L]
    return batch


def occurrence_counts(x: torch.Tensor) -> torch.Tensor:
    """(B, T) long: how many times the token at each position has occurred so far,
    counting the position itself.

    Loop-free: sort each row (stably, so ties keep their order), and within each
    block of equal tokens the occurrence number is the offset from the block's
    first element.
    """
    B, T = x.shape
    pos = torch.arange(T, device=x.device).expand(B, -1)

    order = x.argsort(dim=1, stable=True)
    x_sorted = x.gather(1, order)

    starts_block = torch.ones_like(x_sorted, dtype=torch.bool)
    starts_block[:, 1:] = x_sorted[:, 1:] != x_sorted[:, :-1]
    block_start = torch.cummax(torch.where(starts_block, pos, -torch.ones_like(pos)), dim=1).values

    counts_sorted = pos - block_start + 1
    return torch.empty_like(x).scatter_(1, order, counts_sorted)


if __name__ == "__main__":
    import time

    B, V, L, K = 4096, 128, 128, 25
    for stats in (True, False):
        batch = generate_icl_batch(B, V, L, K, stats=stats)  # warm up
        t0 = time.perf_counter()
        for _ in range(5):
            batch = generate_icl_batch(B, V, L, K, stats=stats)
        dt = (time.perf_counter() - t0) / 5
        nbytes = sum(v.numel() * v.element_size() for v in batch.values())
        print(f"stats={stats!s:5} {dt * 1000:7.1f} ms/batch  {nbytes / 1e6:6.1f} MB  keys={list(batch)}")

    batch = generate_icl_batch(B, V, L, K)
    seq, is_trigg, counts = batch["sequence"], batch["is_trigg"], batch["counts"]
    b_idx, t_idx = torch.where(is_trigg)
    follows_rule = seq[b_idx, t_idx + 1] == batch["output_set"][b_idx, seq[b_idx, t_idx]]
    print(f"trigger -> output rule holds at {follows_rule.float().mean():.1%} of trigger positions")
    print(f"induction possible at {(is_trigg & (counts > 1)).float().mean():.1%} of positions")
