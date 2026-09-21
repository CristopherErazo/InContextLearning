import os
import random
import numpy as np
import torch


MATMUL_PRECISIONS = ("highest", "high", "medium")


def set_matmul_precision(precision: str = "highest") -> str:
    """Chooses how float32 matmuls are computed, returning a line for the log.

    "highest" is true float32 and is PyTorch's default. "high" lets an Ampere or
    newer GPU run the matmul on its tensor cores in TF32: the 8-bit exponent is
    untouched, so the representable range is identical and nothing can overflow
    that would not before, but each input's mantissa is rounded from 24 bits to
    11 (relative error ~5e-4 instead of ~6e-8) while the accumulation stays
    float32. On an A100 that lifts the matmul ceiling from 19.5 to 156 TFLOPS.
    "medium" goes further and rounds the inputs to bfloat16.

    This is a numerical choice, not only a speed one, so it lives in the config
    (`extra_args.matmul_precision`) and is saved with every run: two runs are
    only comparable at the same setting. It affects matmul inputs alone --
    storage, the optimizer and every elementwise op stay float32 -- so it is not
    the same thing as autocast / mixed precision and needs no loss scaling.

    Worth care in this project specifically: most parameters stay frozen at
    random init, so the learned signal is a small quantity emerging against a
    large random background, and sums with heavy cancellation lose far more
    precision than their inputs do. Validate T* against "highest" across the
    d_model sweep before trusting it -- TF32's error grows with the accumulation
    length, i.e. with d_model, so a bias would land straight on the exponent
    being measured. See shell/tf32_ab.sh.
    """
    if precision not in MATMUL_PRECISIONS:
        raise ValueError(f"matmul_precision must be one of {MATMUL_PRECISIONS}, got {precision!r}")
    torch.set_float32_matmul_precision(precision)
    if not torch.cuda.is_available():
        return f"float32 matmul precision = {precision!r} (no CUDA device; CPU math is unaffected)"
    return (f"float32 matmul precision = {precision!r} "
            f"(tf32 {'ON' if precision != 'highest' else 'off'} for matmuls)")


def set_seed(seed: int | None = None) -> int:
    """Sets seeds across Python, NumPy, and PyTorch.

    If seed is None, no seeds are set (runs non-deterministically).
    """
    message = f"Using random seed {seed}."
    if seed is None:
        seed = int(os.environ.get("PYTHONHASHSEED", random.randint(0, 2**32 - 1)))
        message = f"No seed provided. Using random seed {seed} from PYTHONHASHSEED or generated randomly."
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Safe even for multi-GPU setups
    return seed, message

