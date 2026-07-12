import torch


def get_spike_vector(vocab_size:int , P_u:torch.Tensor) -> torch.Tensor:    
    u = torch.empty_like(P_u)
    s_pos = 0.0
    s_neg = 0.0
    for i in range(vocab_size):
        if s_pos <= s_neg:
            u[i] = 1.0
            s_pos += P_u[i].item()
        else:
            u[i] = -1.0
            s_neg += P_u[i].item()
    return u


# def get_distributions(vocab_size:int,b_type:str='dirichlet',alpha:float=1.0,device:str='cpu',u_type:str=None, beta:float=None) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
def get_distributions(args, vocab_size:int, device:str='cpu') -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Get probability distributions P_b, P_u, P_o and P_t for the dual task data generation.

    Args: 
    - args: an object containing the following attributes:

        - b_type (str): Type of bigram distribution to generate. Options are 'dirichlet' and 'spiked'. 'dirichlet' samples the bigram distribution from a dirichlet distribution with concentration parameter alpha. 'spiked' generates a bigram distribution that is correlated with the unigram distribution, with more frequent tokens having higher probabilities of being followed by other tokens, using the formula P_b[i,j] = P_u[j] * (1 - beta * u[i] * u[j]) where u[i] is 1 for more frequent tokens and -1 for less frequent tokens.
        - alpha_d (float): Concentration parameter for the Dirichlet distribution used to generate the big
        - alpha_z (float): Exponent for the Zipf distribution used to generate the
        unigram distribution P_u if b_type is 'dirichlet', or exponent for the zipf distribution if u_type is 'zipf' and b_type is 'spiked'.
        - u_type (str): Type of unigram distribution to generate if b_type is 'spiked'. Options are 'dirichlet' and 'zipf'. If 'dirichlet', the unigram distribution is sampled from a dirichlet distribution with concentration parameter alpha. If 'zipf', the unigram distribution is generated using a zipf distribution with exponent alpha.
        - beta (float): Concentration parameter for the Dirichlet distribution used to generate the bigram distribution P_b from the unigram distribution P_u using the formula P_b[i,j] = P_u[j] * (1 - beta * u[i] * u[j]) where u[i] is 1 for more frequent tokens and -1 for less frequent tokens. This creates a bigram distribution that is correlated with the unigram distribution, with more frequent tokens having higher probabilities of being followed by other tokens. Only used if b_type is 'spiked'.
    
    - vocab_size (int): Size of the vocabulary
    - device (str): Device to create the tensors on
    Returns:
        - P_b (torch.tensor): Bigram probability matrix of shape (V,V) P_b[i,j]  = P(token j | token i)
        - P_u (torch.tensor): Unigram probability vector of shape (V,) P_u[i] = P(token i)
        - P_o (torch.tensor): Probability for output tokens of a trigger token, shape (V,V) P_o[i,j] = P(output token j | trigger token i)
        - P_t (torch.tensor): Probability for trigger tokens, shape (V,) P_t[i] = P(trigger token i)
    """

    if args.b_type == 'dirichlet': 
        print("Sampling bigram distribution from a Dirichlet distribution with concentration parameter alpha_d = ", args.alpha_d)
        # Sample bigram distribution from a dirichlet distribution with concentration parameter alpha
        concentration = torch.ones((vocab_size, vocab_size)) * args.alpha_d
        P_b = torch.distributions.dirichlet.Dirichlet(concentration).sample()
        # Compute unigram distribution as the stationary distribution of P_b
        eigenvalues, eigenvectors = torch.linalg.eig(P_b.T)
        P_u = eigenvectors[:, torch.isclose(eigenvalues.real, torch.tensor(1.0))].real.squeeze()
        P_u /= P_u.sum()  # Normalize
        # Sort P_u and P_b in descending order of P_u for better interpretability (most frequent tokens first)
        sorted_indices = torch.argsort(P_u, descending=True)
        P_u = P_u[sorted_indices]
        P_b = P_b[sorted_indices][:, sorted_indices]

    elif args.b_type == 'spiked':
        print("Generating spiked bigram distribution with and beta = ", args.beta)
        assert args.u_type is not None and args.beta is not None, "u_type and beta must be specified for spiked bigram distribution."
        if args.u_type == 'dirichlet':
            print("Sampling unigram distribution from a Dirichlet distribution with concentration parameter alpha = ", args.alpha_d)
            P_u = torch.distributions.dirichlet.Dirichlet(torch.ones(vocab_size) * args.alpha_d).sample()
        elif args.u_type == 'zipf':
            print("Generating unigram distribution from a Zipf distribution with exponent alpha_z = ", args.alpha_z)
            ranks = torch.arange(1, vocab_size + 1)
            P_u = 1.0 / ranks**args.alpha_z
            P_u /= P_u.sum() # Normalize to get a valid probability distribution
        elif args.u_type == 'uniform':
            print("Generating unigram distribution from a uniform distribution")
            P_u = torch.ones(vocab_size) / vocab_size
        else:
            raise ValueError("Invalid u_type. Options are 'dirichlet' and 'zipf'.")
        
        P_u = P_u.sort(descending=True).values # Sort unigram distribution in descending order for better interpretability (most frequent tokens first)
        u = get_spike_vector(vocab_size, P_u)
        P_b = P_u[None,:] * (1 - args.beta * u[:,None] * u[None,:]) # Generate bigram distribution P_b from unigram distribution P_u using the formula P_b[i,j] = P_u[j] * (1 - beta * u[i] * u[j]) where u[i] is 1 for more frequent tokens and -1 for less frequent tokens. This creates a bigram distribution that is correlated with the unigram distribution, with more frequent tokens having higher probabilities of being followed by other tokens.
        
    # Renormalize
    P_b /= P_b.sum(dim=-1, keepdim=True)    

    assert torch.all(P_b >= 0), "P_b contains negative probabilities."
    assert torch.allclose(P_b.sum(dim=-1), torch.ones(vocab_size)), "Rows of P_b do not sum to 1."
    assert torch.all(P_u >= 0), "P_u contains negative probabilities."
    assert torch.isclose(P_u.sum(), torch.tensor(1.0)), "P_u does not sum to 1."    
    
    P_t = P_u.clone() # Trigger distribution is the same as unigram distribution
    P_o = torch.ones((vocab_size,vocab_size)) / (vocab_size-1) # Uniform distribution over outputs given triggers apart from the trigger token itself
    P_o.fill_diagonal_(0) # Set diagonal to 0 since output cannot be the same as the trigger token
    

    # return P_b.to(device), P_u.to(device), P_o.to(device), P_t.to(device)
    # Return a dictionary of the distributions for better readability
    return {
        "P_b": P_b.to(device),
        "P_u": P_u.to(device),
        "P_o": P_o.to(device),
        "P_t": P_t.to(device)
    }
    

def get_triggers(args,P_t:torch.Tensor):
    """
    Get trigger tokens based on the trigger distribution P_t and the specified method in args (either fixed or random triggers, and if fixed, whether to use the most frequent, least frequent or random tokens as triggers). Returns a tensor of trigger token indices.
    
    Args:

    - args: an object containing the following attributes:
        - fix_trig (bool): Whether to use fixed trigger tokens across all sequences or to sample trigger tokens randomly for each sequence. If True, the same set of trigger tokens will be used for all sequences. If False, trigger tokens will be randomly sampled for each sequence.
        - trig_type (str): If fix_trig is True, this specifies the method for selecting the fixed trigger tokens. Options are 'freq' for the most frequent tokens according to P_t, 'rare' for the least frequent tokens according to P_t, and 'rand' for random tokens sampled according to P_t.
    - P_t: A tensor of shape (V,) representing the trigger token distribution, where P_t[i] is the probability of token i being a trigger token.    
    """
    vocab_size = P_t.shape[0]
    trigger_set = None
    if args.fix_trig:
        if args.trig_type == 'freq':
            trigger_set = [k for k in range(args.K)]
        elif args.trig_type == 'rare':
            trigger_set = [vocab_size - 1 - k for k in range(args.K)] 
        elif args.trig_type == 'rand':
            trigger_set = torch.multinomial(P_t, num_samples=args.K, replacement=False).tolist()
        print(f"Using fixed {args.trig_type} trigger set")
        print(f"Length of trigger set: {len(trigger_set)}, Trigger set: {trigger_set}")
    else:
        print(f"Using random trigger set at each sequence")
    return torch.tensor(trigger_set)


def generate_dual_task_batch(num_samples: int,
                            L: int,
                            K: int,
                            distributions: dict[str, torch.Tensor],
                            trigger_set: torch.Tensor | None = None,
                            device: str | torch.device = "cpu"
                        ) -> dict[str, torch.Tensor]:
    """
    Fully vectorized batch generator for the dual task.

    Returns:
        dict with:
            sequence     : (B, L)
            trigger_set  : (B, K)
            output_set   : (B, K)
            counts       : (B, L)
            is_trigg     : (B, L)
    """

    B = num_samples
    

    # Move to device
    P_b = distributions['P_b'].to(device)
    P_u = distributions['P_u'].to(device)
    P_o = distributions['P_o'].to(device)
    P_t = distributions['P_t'].to(device)
    V = P_u.shape[0]

    # assert P_b.device == 'cuda' , "P_b is not on the correct device, got device = " + str(P_b.device)
    # ---- sample trigger sets ----
    if trigger_set is None:
        trigger_sets = torch.stack([
            torch.multinomial(P_t, K, replacement=False)
            for _ in range(B)
        ], dim=0)
    else:
        # trigger_set = torch.tensor(trigger_set, device=device)
        trigger_set = trigger_set.to(device)
        assert trigger_set.numel() == K
        trigger_sets = trigger_set.unsqueeze(0).repeat(B, 1)

   
    # ---- sample output tokens ----
    # shape: (B, K)
    output_sets = torch.zeros(B, K, dtype=torch.long, device=device)
    for k in range(K):
        output_sets[:, k] = torch.multinomial(
            P_o[trigger_sets[:, k]], 1
        ).squeeze(-1)


    # ---- build trigger mask ----
    trigger_mask = torch.zeros(B, V, dtype=torch.bool, device=device)
    trigger_mask.scatter_(1, trigger_sets, True)

    # ---- build mapping trigger -> output ----
    mapping = torch.full((B, V), -1, dtype=torch.long, device=device)
    mapping.scatter_(1, trigger_sets, output_sets)

    # ---- initialize sequence ----
    sequence = torch.zeros(B, L+1, dtype=torch.long, device=device)
    sequence[:, 0] = torch.multinomial(P_u, B, replacement=True)

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
        prev = current

        # sample bigram transitions
        next_tokens = torch.multinomial(P_b[prev], 1).squeeze(-1)

        # override if trigger
        mapped = mapping[
            torch.arange(B, device=device), prev
        ]

        trigger_positions = mapped != -1

        
        next_tokens[trigger_positions] = mapped[trigger_positions]

        sequence[:, t + 1] = next_tokens

    mask = torch.tril(torch.ones((L, L), dtype=torch.bool),diagonal=-1) # (1, L, L) for multi-head attention
    batch_mask = mask.unsqueeze(0).expand(B, -1, -1) # (batch_size, seq_len, seq_len)
    
    return {
        "sequence": sequence, # shape (B, L+1)
        "trigger_set": trigger_sets, # shape (B, K)
        "output_set": output_sets, # shape (B, K)
        "counts": counts[:,:L],    # shape (B, L)
        "is_trigg": is_trigg[:,:L], # shape (B, L)
        "mask": batch_mask # shape (B, L, L)
    }

