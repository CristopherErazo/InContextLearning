from dataclasses import dataclass, field

from omegaconf import OmegaConf


ui_metadata = {"metadata" : {"ui":True}} # Used in the fields that we want to display in the launch panel as controls

@dataclass
class ModelArgs:
    vocab_size: int = 128 # Vocabulary size
    d_model: int = 256 # Model dimension
    seq_len: int = 128 # Sequence length
    lin_attn: bool = True  # Whether to use linear attention or not
    beta: float = 0.25  # Scaling factor for output logits
    sigma_0: float = 1.0  # Initial std dev for parameter initialization
    pred_mode: str = "next"  # Prediction mode: "last" or "next"
    dropout: float = 0.0  # Dropout rate


@dataclass
class DataArgs:
    batch_size: int = 512  # Batch size for training
    test_size: int = 128  # Number of samples in test set
    rho: float = 0.2  # Fraction of trigger tokens = K/vocab_size
    # Computed Values as placeholders; will be computed later
    K: int = field(default=0)  # Number of trigger tokens
    


@dataclass
class OptimArgs:
    alpha_lr: float = 5.0  # Base learning rate factor
    opt_name: str = "sgd"  # Optimizer type: "adam" or "sgd"
    momentum: float = 0.0  # Momentum for SGD
    weight_decay: float = 0.0  # L2 regularization
    # Computed Values as placeholders; will be computed later
    lr: float = field(default=0)  # Scaled learning rate




@dataclass
class ExtraArgs:
    # alpha_steps: float = 10 # n steps = alpha_steps * vocab_size^2 * sqrt(seq_len)
    # total_steps: int = field(default=0) # Total training steps
    total_steps: int = 5000
    n_prints: int = 150  # Metric evaluation frequency
    n_prints_model: int = 0  # Model checkpoint frequency
    print_scale: str = 'linear'  # Scale for evaluation: log or linear
    experiment_name: str | None  = 'results_test' # Experiment tracking name
    seed: int | None = 42      # Random seed
    enable_control: bool = False  # Enable control features
    enable_rewind: bool = False  # Enable rewind features
    track_artifacts: bool = False
    base_dir : str | None = "./data"  # Base directory for data storage
    launch_token: str | None = None  # Set by rewind.RunLauncher (dashboard); used for the run_id handshake
   


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
    # cfg.extra_args.total_steps = int(cfg.extra_args.alpha_steps * cfg.model_args.vocab_size**2 *math.sqrt(cfg.model_args.seq_len)/cfg.data_args.batch_size)
    return cfg


def load_config(argv: list[str] | None = None) -> TrainerArgs:
    """Defaults merged with `key=value` CLI overrides, derived fields computed.

    `argv` defaults to `sys.argv[1:]` (OmegaConf's own behaviour). Both
    scripts/launcher.py and scripts/train.py start from this.
    """
    defaults = OmegaConf.structured(TrainerArgs())
    overrides = OmegaConf.from_cli(argv) if argv is not None else OmegaConf.from_cli()
    return compute_derived_args(OmegaConf.merge(defaults, overrides))
