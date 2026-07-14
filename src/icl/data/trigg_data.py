import torch

def generate_icl_task_batch(num_samples: int,
                            V: int,
                            L: int,
                            K: int,
                            device: str | torch.device = "cpu"
                        ) -> dict[str, torch.Tensor]:
    """
    Batch generator for the trigger-retrieval task.

    Returns:
        dict with:
            sequence     : (B, L)
            trigger_set  : (B, K)
            output_set   : (B, K)
            counts       : (B, L)
            is_trigg     : (B, L)
    """

    B = num_samples

    if K > V:
        raise ValueError("K must be <= V")

    # Fixed trigger tokens shared across the batch.
    trigger_sets = torch.arange(K, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

    # ---- sample output tokens ----
    # shape: (B, K)
    # Sample K unique tokens from the vocabulary for each sample in the batch,
    # restricted to [K, V-1] so they do not overlap with the fixed trigger tokens.
    output_sets = torch.multinomial(torch.ones(B, V - K, device=device), K, replacement=False) + K

    # ---- build trigger mask ----
    trigger_mask = torch.zeros(B, V, dtype=torch.bool, device=device)
    trigger_mask.scatter_(1, trigger_sets, True)

    # ---- build mapping trigger -> output ----
    mapping = torch.full((B, V), -1, dtype=torch.long, device=device)
    mapping.scatter_(1, trigger_sets, output_sets)

    # ---- initialize sequence ----
    sequence = torch.zeros(B, L+1, dtype=torch.long, device=device)
    sequence[:, 0] = torch.randint(0, V, (B,), device=device)

    # ---- outputs ----
    is_trigg = torch.zeros(B, L+1, dtype=torch.long, device=device)
    counts = torch.zeros(B, L+1, dtype=torch.long, device=device)

    # for counting occurrences
    token_counts = torch.zeros(B, V, dtype=torch.long, device=device)
    # ---- main loop over sequence length ----
    for t in range(L+1):
        current = sequence[:, t]

        # update counts
        token_counts.scatter_add_(
            1,
            current.unsqueeze(1),
            torch.ones(B, 1, dtype=torch.long, device=device),
        )
        counts[:, t] = token_counts[
            torch.arange(B, device=device), current
        ]

        # mark trigger
        is_trigg[:, t] = trigger_mask[
            torch.arange(B, device=device), current
        ].long()

        if t == L:
            break

        # ---- next token ----
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

