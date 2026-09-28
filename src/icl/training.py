"""Training-side helpers shared by scripts/launcher.py and scripts/train.py:
the model / optimizer factories and the training loss."""
from __future__ import annotations

import math

import torch

from .model import MinimalTransformer
from .reduced import ReducedSGD, ReducedTransformer

BACKENDS = ("full", "reduced")
REDUCED_INITS = ("full", "sample")


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
    logits = model(sequence[:, :-1])             # (B, L, V) or (B, 1, V); each layer masks itself
    target = sequence[:, 1:] if model.pred_mode == "next" else sequence[:, -1:]
    return loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))


def build_model(model_args, optim_args, device) -> tuple[torch.nn.Module, torch.optim.Optimizer, str]:
    """Model and optimizer for `model_args.backend`, plus a line for the log.

    "full" is `MinimalTransformer` + `get_optimizer`, exactly as before. "reduced"
    is `ReducedTransformer` + `ReducedSGD` (SGD only, lin_attn only), initialised
    from a composed `MinimalTransformer` (`init="full"`: the same random draw a
    full run with this seed makes) or sampled directly (`init="sample"`, which
    also allows `infinite_d`).
    """
    backend = getattr(model_args, "backend", "full")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    infinite_d = getattr(model_args, "infinite_d", False)

    if backend == "full":
        if infinite_d:
            raise ValueError("infinite_d needs backend='reduced' and init='sample'")
        model = MinimalTransformer(model_args).to(device)
        model.initialize_model()
        optimizer, opt_msg = get_optimizer((p for p in model.parameters() if p.requires_grad), optim_args)
        return model, optimizer, f"backend=full | {opt_msg}"

    init = getattr(model_args, "init", "full")
    if init not in REDUCED_INITS:
        raise ValueError(f"init must be one of {REDUCED_INITS}, got {init!r}")
    if optim_args.opt_name.lower() != "sgd":
        raise ValueError("the reduced backend is exact for SGD only; use backend='full' for "
                         f"opt_name={optim_args.opt_name!r}")
    if init == "full":
        if infinite_d:
            raise ValueError("infinite_d needs init='sample'")
        # Built on `device` and initialised there, like a full run: same RNG draws.
        full = MinimalTransformer(model_args).to(device)
        full.initialize_model()
        model = ReducedTransformer.from_full(full)
        del full
    else:
        d = math.inf if infinite_d else model_args.d_model
        model = ReducedTransformer.sample(model_args, d, model_args.sigma_0, device=device)
    # eta_0 = lr * d: the step size in the normalised (m, q, g) coordinates.
    optimizer = ReducedSGD(model, lr=optim_args.alpha_lr, momentum=optim_args.momentum,
                           weight_decay=optim_args.weight_decay)
    d_msg = "inf" if math.isinf(model.d_model) else model.d_model
    return model, optimizer, (f"backend=reduced (init={init}, d={d_msg}) | "
                              f"ReducedSGD, eta_0 = alpha_lr = {optim_args.alpha_lr}")
