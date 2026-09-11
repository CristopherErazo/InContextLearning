"""icl: a minimal two-layer linear-attention transformer trained on the
trigger-retrieval task, plus the evaluation code used to study induction heads.

The package lives in `src/` and is installed under the name `icl`
(see `[tool.hatch.build.targets.wheel.sources]` in pyproject.toml).
"""

from icl.config import TrainerArgs, ModelArgs, DataArgs, OptimArgs, ExtraArgs, compute_derived_args
from icl.data import generate_icl_batch
from icl.model import MinimalTransformer
from icl.utils import set_seed
from icl.evaluation import *  # noqa: F401,F403  (re-exports evaluation.__all__)
from icl.evaluation import __all__ as _evaluation_all

__all__ = [
    "TrainerArgs",
    "ModelArgs",
    "DataArgs",
    "OptimArgs",
    "ExtraArgs",
    "compute_derived_args",
    "generate_icl_batch",
    "MinimalTransformer",
    "set_seed",
    *_evaluation_all,
]
