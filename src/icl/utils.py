import os
import random
import numpy as np
import torch


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

