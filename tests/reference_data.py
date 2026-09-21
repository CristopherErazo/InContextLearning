"""The pre-refactor batch generator, kept verbatim as a reference.

`icl.data.generate_icl_batch` replaced this loop-based version with a
vectorised one. It is preserved here (and nowhere in the package) so that
`test_data.py` can check the two agree distributionally and
`scripts/bench_data.py` can time them against each other. It is not imported by
any production code and should not grow features.
"""
from __future__ import annotations

import torch


def generate_icl_batch_loop(num_samples: int,
                            V: int,
                            L: int,
                            K: int,
                            device: str | torch.device = "cpu",
                            ) -> dict[str, torch.Tensor]:
    """Trigger-retrieval batch, one Python iteration per sequence position."""
    B = num_samples

    if K > V:
        raise ValueError("K must be <= V")

    trigger_sets = torch.arange(K, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

    output_sets = torch.multinomial(torch.ones(B, V - K, device=device), K, replacement=False) + K

    trigger_mask = torch.zeros(B, V, dtype=torch.bool, device=device)
    trigger_mask.scatter_(1, trigger_sets, True)

    mapping = torch.full((B, V), -1, dtype=torch.long, device=device)
    mapping.scatter_(1, trigger_sets, output_sets)

    sequence = torch.zeros(B, L + 1, dtype=torch.long, device=device)
    prob_dist = torch.ones(B, V, device=device)
    prob_dist[mapping != -1] = 2
    prob_dist /= prob_dist.sum(dim=1, keepdim=True)
    sequence[:, 0] = torch.multinomial(prob_dist, 1).squeeze(1)

    is_trigg = torch.zeros(B, L + 1, dtype=torch.long, device=device)
    counts = torch.zeros(B, L + 1, dtype=torch.long, device=device)

    token_counts = torch.zeros(B, V, dtype=torch.long, device=device)
    for t in range(L + 1):
        current = sequence[:, t]

        token_counts.scatter_add_(
            1,
            current.unsqueeze(1),
            torch.ones(B, 1, dtype=torch.long, device=device),
        )
        counts[:, t] = token_counts[torch.arange(B, device=device), current]

        is_trigg[:, t] = trigger_mask[torch.arange(B, device=device), current].long()

        if t == L:
            break

        next_tokens = torch.randint(0, V, (B,), device=device)
        mapped = mapping[torch.arange(B, device=device), current]
        trigger_positions = mapped != -1
        next_tokens[trigger_positions] = mapped[trigger_positions]

        sequence[:, t + 1] = next_tokens

    mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device), diagonal=-1)
    batch_mask = mask.unsqueeze(0).expand(B, -1, -1)

    return {
        "sequence": sequence,
        "trigger_set": trigger_sets,
        "output_set": output_sets,
        "counts": counts[:, :L],
        "is_trigg": is_trigg[:, :L],
        "mask": batch_mask,
    }
