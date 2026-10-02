"""icl: a minimal two-layer linear-attention transformer trained on the
trigger-retrieval task, plus the evaluation code used to study induction heads.

Standard src layout: the package body is `src/icl/`.
"""

from icl.config import TrainerArgs, ModelArgs, DataArgs, OptimArgs, ExtraArgs, compute_derived_args, load_config
from icl.data import generate_icl_batch
from icl.model import MinimalTransformer, make_mask
from icl.utils import set_matmul_precision, set_seed
from icl.reduced import ReducedSGD, ReducedTransformer
from icl.training import build_model, get_optimizer, compute_loss
from icl.runs import RunData
from icl.evaluation import *  # noqa: F401,F403  (re-exports evaluation.__all__)
from icl.evaluation import __all__ as _evaluation_all
from icl.theory import *  # noqa: F401,F403  (re-exports theory.__all__)
from icl.theory import __all__ as _theory_all

__all__ = [
    "TrainerArgs", "ModelArgs", "DataArgs", "OptimArgs", "ExtraArgs",
    "compute_derived_args", "load_config",
    "generate_icl_batch",
    "MinimalTransformer", "make_mask",
    "set_seed", "set_matmul_precision",
    "ReducedTransformer", "ReducedSGD",
    "build_model", "get_optimizer", "compute_loss",
    "RunData",
    *_evaluation_all,
    *_theory_all,
]
