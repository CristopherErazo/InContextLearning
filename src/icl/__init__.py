"""icl: a minimal two-layer linear-attention transformer trained on the
trigger-retrieval task, plus the evaluation code used to study induction heads.

Standard src layout: the package body is `src/icl/`.
"""

from icl.config import TrainerArgs, ModelArgs, DataArgs, OptimArgs, ExtraArgs, compute_derived_args, load_config
from icl.data import generate_icl_batch
from icl.model import MinimalTransformer
from icl.utils import set_seed
from icl.training import get_optimizer, compute_loss
from icl.evaluation import *  # noqa: F401,F403  (re-exports evaluation.__all__)
from icl.evaluation import __all__ as _evaluation_all

__all__ = [
    "TrainerArgs", "ModelArgs", "DataArgs", "OptimArgs", "ExtraArgs",
    "compute_derived_args", "load_config",
    "generate_icl_batch",
    "MinimalTransformer",
    "set_seed",
    "get_optimizer", "compute_loss",
    *_evaluation_all,
]
