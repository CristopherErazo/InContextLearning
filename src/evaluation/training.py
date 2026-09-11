import torch

def get_optimizer(trainable_params,args):
    """
    Get the optimizer based on the configuration in args.
    """
    opt_name = args.opt_name.lower()
    lr = args.lr
    weight_decay = args.weight_decay
    momentum = args.momentum
    
    kwargs = {'lr': lr,'weight_decay': weight_decay}
    if opt_name == 'sgd':
        message = "Using SGD optimizer"
        kwargs['momentum'] = momentum
        optimizer = torch.optim.SGD(trainable_params, **kwargs)
    elif opt_name == 'adam':
        message = "Using Adam optimizer"
        optimizer = torch.optim.Adam(trainable_params, **kwargs)
    elif opt_name == 'adamw':
        message = "Using AdamW optimizer"
        optimizer = torch.optim.AdamW(trainable_params, **kwargs)
    else:
        raise ValueError("Invalid optimizer type. Options are 'SGD', 'adam', and 'adamW'.")
    return optimizer, message   


def evaluate_model(model, batch, loss_fn, device):
    """
    Evaluate the minimal model on the given batch and return the computed loss.
    This function is used for evaluation during training.
    """
    # Extract evaluation mode from the model
    eval_mode = model.pred_mode  # 'last' or 'next' for next-token prediction or last-token prediction
    # Evaluate model on the dual
    sequence = batch['sequence'].to(device)  # shape (batch_size, seq_len + 1)
    input = sequence[:, :-1]  # shape (batch_size, seq_len)
    mask = batch['mask'].to(device)  # shape (batch_size, seq_len, seq_len)
    if eval_mode == 'last':
        target = sequence[:, -1:]  # shape (batch_size, 1)
    elif eval_mode == 'next':
        target = sequence[:, 1:]  # shape (batch_size, seq_len)
    else:
        raise ValueError("Invalid evaluation mode. Options are 'last' or 'next'.")

    logits = model(input, mask)  # shape (batch_size, seq_len, vocab_size) or (batch_size, 1, vocab_size) if eval_mode == 'last'

    # Reshape logits and target for loss computation
    if eval_mode == 'last':
        logits_masked = logits[:, -1, :]  # shape (batch_size, vocab_size)
        target_masked = target[:, -1]  # shape (batch_size,)
    else:
        logits_masked = logits.reshape(-1, logits.size(-1))  # shape (batch_size * seq_len, vocab_size)
        target_masked = target.reshape(-1)  # shape (batch_size * seq_len,)

    # Compute loss
    # print(f"logits_mean: {logits_masked.mean().item():.4f}, logits_std: {logits_masked.std().item():.4f}, target_mean: {target_masked.float().mean().item():.4f}")
    loss = loss_fn(logits_masked, target_masked)
    return loss