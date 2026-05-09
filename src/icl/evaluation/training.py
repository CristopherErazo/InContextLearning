import torch


def evaluate_model(model,batch,loss_fn,path,device):
    """ 
    Evaluate the model on the given batch and return the computed loss. This function is used for evaluation during training.    
    """
    # Evaluate model on the dual 
    sequence = batch['sequence'].to(device) # shape (batch_size, seq_len + 1)
    input = sequence[:, :-1] # shape (batch_size, seq_len)
    target = sequence[:, 1:] # shape (batch_size, seq_len)
    mask = batch['mask'].to(device) # shape (batch_size, seq_len, seq_len)
    counts = batch['counts'].to(device) # shape (batch_size, seq_len)
    trigg_set = batch['trigger_set'].to(device) # shape (batch_size, K)

    logits = model(input, mask, path='full' if path=='full_trigg' else path) # shape (batch_size, seq_len, vocab_size)
    
    if path == 'full':
        all = torch.ones_like(input, dtype=torch.bool) # shape (B, L)
        logits_masked = logits[all] # shape (num_masked_positions, vocab_size)
        target_masked = target[all] # shape (num_masked_positions,)
    elif path == 'bigram':
        only_non_triggers = ~( (input.unsqueeze(-1) == trigg_set.unsqueeze(1)).any(-1) ) # shape (B, L)
        logits_masked = logits[only_non_triggers] # shape (num_masked_positions, vocab_size)
        target_masked = target[only_non_triggers] # shape (num_masked_positions,)
    elif path == 'induction':
        # only_triggers = (input.unsqueeze(-1) == trigg_set.unsqueeze(1)).any(-1) & (counts >= 2) # shape (B, L)
        only_triggers = torch.ones_like(input, dtype=torch.bool) # shape (B, L)
        logits_masked = logits[only_triggers] # shape (num_masked_positions, vocab_size)
        target_masked = target[only_triggers] # shape (num_masked_positions,)
    elif path == 'full_trigg':
        only_triggers = (input.unsqueeze(-1) == trigg_set.unsqueeze(1)).any(-1) & (counts >= 2) # shape (B, L)
        logits_masked = logits[only_triggers] # shape (num_masked_positions, vocab_size)
        target_masked = target[only_triggers] # shape (num_masked_positions,)
    else:
        raise ValueError("Invalid path type. Options are 'full', 'bigram', 'induction', 'full_trigg'.")

    # Compute loss and update model
    loss_trigg = loss_fn(logits_masked, target_masked)
    return loss_trigg


