from pathlib import Path
from dataclasses import dataclass, field


ui_metadata = {"metadata" : {"ui":True}} # Used in the fields that we want to display in the launch panel as controls

@dataclass
class ModelArgs:
    vocab_size: int = 150 # Vocabulary size
    d_model: int = 150 # Model dimension
    seq_len: int = 125 # Sequence length
    lin_attn: bool = True  # Whether to use linear attention or not
    beta: float = 0.25  # Scaling factor for output logits
    sigma_0: float = 1.0  # Initial std dev for parameter initialization
    pred_mode: str = "next"  # Prediction mode: "last" or "next"
    dropout: float = 0.0  # Dropout rate


@dataclass
class DataArgs:
    batch_size: int = 512  # Batch size for training
    test_size: int = 512  # Number of samples in test set
    rho: float = 0.2  # Fraction of trigger tokens = K/vocab_size
    # Computed Values as placeholders; will be computed later
    K: int = field(default=0)  # Number of trigger tokens
    


@dataclass
class OptimArgs:
    alpha_lr: float = 0.0005  # Base learning rate factor
    opt_name: str = "adam"  # Optimizer type: "adam" or "sgd"
    momentum: float = 0.9  # Momentum for SGD
    weight_decay: float = 0.0  # L2 regularization
    # Computed Values as placeholders; will be computed later
    lr: float = field(default=0)  # Scaled learning rate




@dataclass
class ExtraArgs:
    total_steps: int = 1000 # Total training steps
    n_prints: int = 150  # Metric evaluation frequency
    n_prints_model: int = 150  # Model checkpoint frequency
    print_scale: str = 'linear'  # Scale for evaluation: log or linear
    experiment_name: str | None  = 'results' # Experiment tracking name
    seed: int | None = 42      # Random seed
    enable_control: bool = False  # Enable control features
    enable_rewind: bool = False  # Enable rewind features
    track_artifacts: bool = True
    base_dir : str | None = "./data"  # Base directory for data storage
   


@dataclass
class TrainerArgs:
    """
    TrainerArgs is a dataclass that encapsulates all the configuration parameters required for training a model.
    It includes model-specific arguments, data handling parameters, optimization settings, and additional configurations.
    """
    model_args: ModelArgs = field(default_factory=ModelArgs)
    data_args: DataArgs = field(default_factory=DataArgs)
    optim_args: OptimArgs = field(default_factory=OptimArgs)
    extra_args: ExtraArgs = field(default_factory=ExtraArgs)


def compute_derived_args(cfg: TrainerArgs) -> TrainerArgs:
    """Recalculates dynamic fields after CLI arguments are merged."""
    cfg.data_args.K = int(cfg.model_args.vocab_size * cfg.data_args.rho)
    cfg.optim_args.lr = cfg.optim_args.alpha_lr * cfg.data_args.batch_size / cfg.model_args.d_model
    return cfg



