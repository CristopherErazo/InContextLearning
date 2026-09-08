import torch

def generate_icl_batch(num_samples: int,
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
    # I want to sample the initial sequence from the stationary distribution of each 
    # sequence which has twice probability of being an OUTPUT token than any other token. 
    # But we need to consider that the outputs at each position are different for each sequence, so we need to sample from a different distribution for each sequence.
    prob_dist = torch.ones(B, V, device=device)
    prob_dist[mapping != -1] = 2
    prob_dist /= prob_dist.sum(dim=1, keepdim=True)
    sequence[:, 0] = torch.multinomial(prob_dist, 1).squeeze(1)

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
        "trigger_set": trigger_sets, # shape (B, K)
        "output_set": output_sets,
        "counts": counts[:, :L],
        "is_trigg": is_trigg[:, :L],
        "mask": batch_mask,
    }


if __name__ == "__main__":
    # Example usage
    B, V, L, K = 5000, 10, 70, 3
    batch = generate_icl_batch(num_samples=B, V=V, L=L, K=K)
    logits = torch.randn(B, L, V)  # shape (B, L, V)

    input = batch["sequence"][:, :-1]  # shape (B, L)
    is_trigg = batch["is_trigg"]  # shape (B, L)
    counts = batch["counts"]  # shape (B, L)
    target = batch["sequence"][:, 1:]  # shape (B, L)

    # Mask for positions where induction is possible (is_trigg == 1 and counts > 1)
    ind_possible = (is_trigg == 1) & (counts > 1) # shape (B, L)

    # Extract the batch and sequence indices of positions where induction is possible
    b_idx, l_idx = torch.where(ind_possible) # shape (num_masked_positions,)

    # Masked logits and targets for positions where induction is possible
    N = b_idx.shape[0]  # number of positions where induction is possible = num_masked_positions
    masked_logits = logits[b_idx, l_idx]  # shape (num_masked_positions, V)
    masked_targets = target[b_idx, l_idx]  # shape (num_masked_positions,)

    # Get on-off target logits for positions where induction is possible
    # Target logits
    on_target_logits = masked_logits.gather( 1, masked_targets[:, None]).squeeze(1)    # (N,)

    # All non-target logits
    token_idx = torch.arange(V, device=logits.device)
    off_mask = token_idx[None, :] != masked_targets[:, None]

    off_target_logits = masked_logits[off_mask].reshape(N, V - 1) # (N, V-1)

    # Create a dictionary to store the results for each sequence length
    results = {
        "on": {},
        "off": {},
    }

    for l in range(L):
        mask = l_idx == l

        results["on"][l] = on_target_logits[mask]
        results["off"][l] = off_target_logits[mask]

        print(f"Sequence length {l}: on = {results['on'][l].shape}, off = {results['off'][l].shape}")



