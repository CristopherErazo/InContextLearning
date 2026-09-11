"""Training-side helpers shared by scripts/launcher.py and scripts/train.py:
the optimizer factory and the training loss."""
from __future__ import annotations

import torch


def get_optimizer(trainable_params, args) -> tuple[torch.optim.Optimizer, str]:
    """Build the optimizer named by `args.opt_name` ('sgd', 'adam', 'adamw')."""
    name = args.opt_name.lower()
    kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if name == "sgd":
        optimizer = torch.optim.SGD(trainable_params, momentum=args.momentum, **kwargs)
    elif name == "adam":
        optimizer = torch.optim.Adam(trainable_params, **kwargs)
    elif name == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, **kwargs)
    else:
        raise ValueError(f"opt_name must be 'sgd', 'adam' or 'adamw', got {args.opt_name!r}")
    return optimizer, f"Using {type(optimizer).__name__} optimizer"


def compute_loss(model, batch: dict, loss_fn, device) -> torch.Tensor:
    """Training loss on one batch: next-token prediction over every position,
    or over the last position only when `model.pred_mode == 'last'`."""
    sequence = batch["sequence"].to(device)      # (B, L+1)
    mask = batch["mask"].to(device)              # (B, L, L)
    logits = model(sequence[:, :-1], mask)       # (B, L, V) or (B, 1, V)
    target = sequence[:, 1:] if model.pred_mode == "next" else sequence[:, -1:]
    return loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
