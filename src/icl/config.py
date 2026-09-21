import math
from dataclasses import dataclass, field

from omegaconf import OmegaConf


ui_metadata = {"metadata" : {"ui":True}} # Used in the fields that we want to display in the launch panel as controls

MIN_TOTAL_STEPS = 100  # lower bound on a budget derived from extra_args.alpha_steps

@dataclass
class ModelArgs:
    vocab_size: int = 128 # Vocabulary size
    d_model: int = 256 # Model dimension
    seq_len: int = 128 # Sequence length
    lin_attn: bool = True  # Whether to use linear attention or not
    beta: float = 0.25  # Scaling factor for output logits
    sigma_0: float = 1.0  # Initial std dev for parameter initialization
    pred_mode: str = "next"  # Prediction mode: "last" or "next"
    mask1: str = "causal"  # Layer-1 mask: "causal" (j < i) or "prev" (j == i-1 only)
    mask2: str = "causal"  # Layer-2 mask: "causal" (j < i) or "prev" (j == i-1 only)
    dropout: float = 0.0  # Dropout rate


@dataclass
class DataArgs:
    alpha_batch: float = 1000.0  # Scaling factor for batch size ~1/(L * frac_solvable)
    batch_size: int | None = field(default=None)  # Batch size (computed from alpha_batch) if given, overrides alpha_batch
    test_size: int = 256  # Number of samples in test set
    rho: float = 0.2  # Fraction of trigger tokens = K/vocab_size
    gen_device: str = "auto"  # Where batches are sampled: "cpu", "cuda", or "auto" (= training device)
    # Computed Values as placeholders; will be computed later
    K: int = field(default=0)  # Number of trigger tokens



@dataclass
class OptimArgs:
    alpha_lr: float = 2500.0  # Base learning rate factor
    opt_name: str = "sgd"  # Optimizer type: "adam" or "sgd"
    momentum: float = 0.0  # Momentum for SGD
    weight_decay: float = 0.0  # L2 regularization
    # Computed Values as placeholders; will be computed later
    lr: float = field(default=0)  # Scaled learning rate




@dataclass
class ExtraArgs:
    # Step budget. `total_steps` is used as given unless `alpha_steps` is set, in
    # which case it is derived from the predicted learning time T* ~ V^2 sqrt(L)
    # (see compute_derived_args). The scaling sweeps set alpha_steps so that every
    # configuration gets the same budget *in units of its own T**.
    alpha_steps: float | None = None  # Scaling factor for total steps
    total_steps: int = field(default=1000)  # Total number of training steps
    stop_at_accuracy: float | None = None  # Stop once top1_accuracy reaches this (None = never)
    n_prints: int = 75  # Metric evaluation frequency
    n_prints_model: int = 75  # Model checkpoint frequency
    print_scale: str = 'linear'  # Scale for evaluation: log or linear
    experiment_name: str | None  = 'icl' # Experiment tracking name
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
    cfg.optim_args.lr = cfg.optim_args.alpha_lr / cfg.model_args.d_model
    lmb = cfg.model_args.seq_len / (cfg.model_args.vocab_size * (1 + cfg.data_args.rho))
    if cfg.data_args.batch_size is None:    
        cfg.data_args.batch_size = max(1, int(cfg.data_args.alpha_batch /(cfg.model_args.seq_len * frac_solvable_positions(cfg.data_args.rho, lmb))))

    if cfg.extra_args.alpha_steps is not None:
        # Floor: at very large eta_0 the prediction is a handful of steps, and a run that
        # short cannot show whether it diverges or merely fails to escape -- which is the
        # distinction shell/eta_scan.sh exists to make. Never binds at sane parameters.
        cfg.extra_args.total_steps = max(MIN_TOTAL_STEPS, int(cfg.extra_args.alpha_steps * predicted_learning_time(cfg)))

    return cfg


def load_config(argv: list[str] | None = None) -> TrainerArgs:
    """Defaults merged with `key=value` CLI overrides, derived fields computed.

    `argv` defaults to `sys.argv[1:]` (OmegaConf's own behaviour). Both
    scripts/launcher.py and scripts/train.py start from this.
    """
    defaults = OmegaConf.structured(TrainerArgs())
    overrides = OmegaConf.from_cli(argv) if argv is not None else OmegaConf.from_cli()
    return compute_derived_args(OmegaConf.merge(defaults, overrides))


def frac_solvable_positions(rho: float, lam:float) -> float:
    """Returns the fraction of positions that are solvable given rho and lambda.

    Args:
        rho (float): Fraction of trigger tokens (K/vocab_size).
        lam (float): Lambda parameter (L/(V*(1+rho)))

    Returns:
        float: Fraction of solvable positions.
    """
    return (rho/(1+rho)) * (1- (1-math.exp(-lam)) / lam)


def predicted_learning_time(cfg) -> float:
    """Theory estimate of T* in gradient steps (paper/scratch/bimodal_derivation.tex).

    T* = C * V^2 sqrt(L) / eta_0,   C = 2(1+rho)^2 sqrt(rho(1-rho)) / (beta rho)
    with eta_0 = lr * d_model, since one SGD step advances gradient-flow time by 1.
    Budgeting only -- it is the quantity the sweeps are measuring, not an input to them.
    """
    V, L, d = cfg.model_args.vocab_size, cfg.model_args.seq_len, cfg.model_args.d_model
    rho = cfg.data_args.K / V          # the realised rho, since K = int(rho*V)
    beta = cfg.model_args.beta
    eta_0 = cfg.optim_args.lr * d      # == alpha_lr today; survives a change of lr convention
    C = 2 * (1 + rho)**2 * math.sqrt(rho * (1 - rho)) / (beta * rho)
    return C * V**2 * math.sqrt(L) / eta_0
