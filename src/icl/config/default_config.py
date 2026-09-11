from pathlib import Path
from dataclasses import dataclass, field


ui_metadata = {"metadata" : {"ui":True}} # Used in the fields that we want to display in the launch panel as controls

@dataclass
class ModelArgs:
    vocab_size: int = field(default=256,**ui_metadata)  # Vocabulary size
    alpha_dim: float = field(default=0.2,**ui_metadata)  # Quotient d/V
    alpha_length: float = field(default=1.0,**ui_metadata)  # Quotient L/V
    lin_attn: bool = True  # Whether to use linear attention or not
    beta: float = 0.5  # Scaling factor for output logits
    sigma_0: float = 1.0  # Initial std dev for parameter initialization
    pred_mode: str = "next"  # Prediction mode: "last" or "next"
    dropout: float = 0.0  # Dropout rate
    # Computed Values as placeholders; will be computed later
    d_model: int = field(default=0)
    seq_len: int = field(default=0)


@dataclass
class DataArgs:
    batch_size: int = 512  # Batch size for training
    test_size: int = 4096  # Number of samples in test set
    rho: float = field(default=0.2,**ui_metadata)  # Fraction of trigger tokens = K/vocab_size
    # Computed Values as placeholders; will be computed later
    K: int = field(default=0)  # Number of trigger tokens
    


@dataclass
class OptimArgs:
    V_base: int = 64  # Base dimension for optimizer scaling
    alpha_lr: float = field(default=0.002,**ui_metadata)  # Base learning rate factor
    opt_name: str = "adam"  # Optimizer type: "adam" or "sgd"
    momentum: float = 0.9  # Momentum for SGD
    weight_decay: float = 0.0  # L2 regularization
    # Computed Values as placeholders; will be computed later
    lr: float = field(default=0)  # Scaled learning rate




@dataclass
class ExtraArgs:
    alpha_steps: float = field(default=35,**ui_metadata)  # Sample complexity = total_steps / V
    n_prints: int = 150  # Metric evaluation frequency
    n_prints_model: int = 150  # Model checkpoint frequency
    print_scale: str = 'linear'  # Scale for evaluation: log or linear
    experiment_name: str | None  = 'tmp' # Experiment tracking name
    comments: str = ''  # Additional run comments
    seed: int | None = 42      # Random seed
    enable_control: bool = field(default=False,**ui_metadata)  # Enable control features
    enable_rewind: bool = field(default=False,**ui_metadata)  # Enable rewind features
    track_artifacts: bool = True
    # Computed Values
    total_steps: int = field(default=0)  # Total training steps

    launch_token: str | None = None
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
    cfg.model_args.d_model = int(cfg.model_args.vocab_size * cfg.model_args.alpha_dim)
    cfg.model_args.seq_len = int(cfg.model_args.d_model * cfg.model_args.alpha_length)
    cfg.data_args.K = int(cfg.model_args.vocab_size * cfg.data_args.rho)
    cfg.extra_args.total_steps = int(cfg.model_args.vocab_size * cfg.extra_args.alpha_steps)
    cfg.optim_args.lr = cfg.optim_args.alpha_lr * cfg.optim_args.V_base / cfg.model_args.vocab_size
    return cfg



