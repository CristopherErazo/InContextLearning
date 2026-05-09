import time
import numpy as np
from omegaconf import OmegaConf
from dataclasses import dataclass , field
from typing import Optional
import torch

from tracklab import ExperimentTracker
from icl import *
from icl.evaluation.utils import get_sub_batch


@dataclass
class ModelArgs:
    vocab_size: int = 32  # Vocabulary size
    seq_len: int = 64 # Sequence length
    d_model: int = 1024 # Model dimension
    rank: int = 16 # rank or matrices
    dropout: float = 0.0 # Dropout rate
    lin_attn: bool = False # Whether to use linear attention or not
    # path: str = "full" # Path to follow (options are "full", "induction" and "bigram")

@dataclass
class DataArgs:
    b_type: str = 'spiked' # P_b distribution type: dirichlet or spiked
    alpha_d: float = 0.1 # Dirichlet concentration parameter for bigram distribution (only used if b_type is dirichlet or u_type is dirichlet)
    alpha_z: Optional[float] = 1.0 # Exponent for the Zipf distribution used to generate the unigram distribution P_u if b_type is 'spiked' and u_type is 'zipf'
    u_type: Optional[str] = 'uniform' # P_u distribution type: dirichlet or zipf (only used if b_type is spiked)
    beta: Optional[float] = 0. # Beta parameter for spiked bigram distribution (only used if b_type is spiked)
    fix_trig: bool = True # Whether to fix the trigger tokens or not
    trig_type: Optional[str] = 'freq' # Type of fixed trigger tokens if fix_trig is True (options are 'freq', 'rare' and 'rand')
    batch_size: int = 64 # Batch size for training
    test_size: int = 200 # Number of samples in the test set
    K : int = 10 # Number of trigger tokens  

@dataclass
class OptimArgs:
    lr: float = 0.0005
    opt: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0

@dataclass 
class ExtraArgs:
    total_steps: int = 3000 # Number of training steps
    n_prints: int = 100 # Number of times to print during training.
    n_prints_model: int = 5 # Number of times to save model checkpoints during training.
    print_scale: str = 'linear' # Scale for printing steps: log or linear
    experiment_name: str = 'tmp_theory' # Name of the experiment for saving results
    file_name: str = 'results' # Name of the file for saving results
    path: str = "induction" # Path to follow (options are "full", "induction" and "bigram")

@dataclass
class TrainerArgs:
    model_args: ModelArgs = field(default_factory=ModelArgs)
    optim_args: OptimArgs = field(default_factory=OptimArgs)
    data_args: DataArgs = field(default_factory=DataArgs)
    extra_args: ExtraArgs = field(default_factory=ExtraArgs)




def main():

    defaults = OmegaConf.structured(TrainerArgs())
    cli_config = OmegaConf.from_cli()
    cfg = OmegaConf.merge(defaults, cli_config)
    
    
    # Create an experiment and run
    experiment_name = cfg.extra_args.experiment_name
    exp = ExperimentTracker(experiment_name)
    run  = exp.start_run(cfg, artifacts=True)

    # Initialize logger for the run
    logger = run.get_logger(log_to_file=True, log_to_terminal=True)

    # Print the parameters
    logger.info("Experiment Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    


    # Get all parameters in a flat dictionary and take out some parameters
    flat_dict = OmegaConf.to_container(cfg, resolve=True)
    vocab_size = cfg.model_args.vocab_size
    seq_len = cfg.model_args.seq_len
    opt_name = cfg.optim_args.opt.lower()
    test_size = cfg.data_args.test_size
    batch_size = cfg.data_args.batch_size
    K = cfg.data_args.K
    path = cfg.extra_args.path


    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    # Define distributions
    distributions = get_distributions(cfg.data_args, vocab_size, device=device)

    # Compute entropies and reference KL divergences and print
    dist_measures = compute_entropies_and_dkl(distributions['P_b'], distributions['P_u'])
    logger.info("\n".join(f"{key}: {value:.4f}" for key, value in dist_measures.items()))

    # Initialize Model
    # model = DualModel(cfg.model_args).to(device)
    model = LowRankTransformer(cfg.model_args).to(device)
    model = initialize_model(model,path=path)

    # Define triggers
    trigger_set = get_triggers(cfg.data_args, distributions['P_t'])#.to(device)

    # Print trainable parameters
    logger.info("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}, {param.shape}")

    # Define loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    kwargs = {'lr': cfg.optim_args.lr,'weight_decay': cfg.optim_args.weight_decay}
    if opt_name == 'sgd':
        logger.info("Using SGD optimizer")
        kwargs['momentum'] = cfg.optim_args.momentum
        optimizer = torch.optim.SGD(trainable_params, **kwargs)
    elif opt_name == 'adam':
        logger.info("Using Adam optimizer")
        optimizer = torch.optim.Adam(trainable_params, **kwargs)
    elif opt_name == 'adamw':
        logger.info("Using AdamW optimizer")
        optimizer = torch.optim.AdamW(trainable_params, **kwargs)
    else:
        raise ValueError("Invalid optimizer type. Options are 'SGD', 'adam', and 'adamW'.")

    test_batch = generate_dual_task_batch(test_size,
                                          seq_len,
                                          K,
                                          distributions,
                                          trigger_set=trigger_set)
                                        #   device=device)
    sub_batch = get_sub_batch(test_batch,device = device, n_test=10)





    opt_losses = optimal_pop_losses(test_batch, P_b=distributions['P_b'])

    logger.info("\n".join(f"{key}: {value:.4f}" for key, value in opt_losses.items()))



    metrics = [
        IC_TopKAccuracy(1),
        IC_TopKAccuracy(3),
        LossMetric(name = "loss",
                   logits_fn = lambda ctx: ctx.logits_induction[ctx.all],
                   target_fn = lambda ctx: ctx.target[ctx.all],
                   rescale = True),
        # LossMetric(name = "loss_bigram",
        #            logits_fn = lambda ctx: ctx.logits_bigram[ctx.only_non_triggers],
        #            target_fn = lambda ctx: ctx.target[ctx.only_non_triggers], 
        #            rescale = False),
        # LossMetric(name = "loss_ind",
        #            logits_fn = lambda ctx: ctx.logits_induction[ctx.only_triggers],
        #            target_fn = lambda ctx: ctx.target[ctx.only_triggers], 
        #            rescale = False),    
        # KLMetric(name="kl_b_total",
        #         P_fn=lambda ctx: ctx.P_b[ctx.input],
        #         Q_fn=lambda ctx: ctx.model_prob,
        #         ),
        # KLMetric(name="kl_b_bigram",
        #         P_fn=lambda ctx: ctx.P_b[ctx.input],
        #         Q_fn=lambda ctx: ctx.model_prob_bigram,
        #         ),             
        M(),
        Gamma(),
        Eta(),
        Q(),
        Sigma1()       
        ]




    evaluator = Evaluator(metrics)


    # Training loop parameters
    total_steps = cfg.extra_args.total_steps
    nprints = cfg.extra_args.n_prints
    nprints_model = cfg.extra_args.n_prints_model
    print_scale = cfg.extra_args.print_scale

    
    if print_scale == 'log':
        print_total_steps = np.unique(np.logspace(-0.01, np.log10(total_steps-1), num=nprints).astype(int))
        print_total_steps_model = np.unique(np.logspace(-0.01, np.log10(total_steps-1), num=nprints_model).astype(int))
    elif print_scale == 'linear':
        print_total_steps = np.linspace(0, total_steps-1, num=nprints).astype(int)
        print_total_steps_model = np.linspace(0, total_steps-1, num=nprints_model).astype(int)

    logger.info(f"Step/{total_steps}\t" + "\t".join(m.name for m in metrics) + "\t loss_eff")
    
    t0 = time.time()

    logger.info("Starting training")
    for step in range(total_steps):
        # Evaluations and logging of scalars
        if step in print_total_steps:
            res = evaluator.evaluate(model, test_batch, loss_fn, distributions['P_b'], distributions['P_u'])
            L_eff = loss_eff(res['m'],
                             res['sigma1'],
                             res['q'],
                             res['eta'],
                             res['gamma'],
                             d=cfg.model_args.d_model,
                             L=cfg.model_args.seq_len,
                             V=cfg.model_args.vocab_size,
                             K=cfg.data_args.K,
                             )
            res['loss_eff'] = L_eff.item()

            run.track_metric(step, **res)
            logger.info(f"{step}\t" + "\t".join(f"{v:.4f}" for v in res.values()))

            
            # res = evaluator.evaluate(model, test_batch, loss_fn, distributions['P_b'], distributions['P_u'])
            # run.track_metric(step, res, split="test")
            
        # Evaluations and logging of attention patterns
        if step in print_total_steps_model:
            att_patterns = get_attention_patterns(model, sub_batch , path, device)
            logger.info(f'Saving attention patterns at step {step} with shapes: ' + ", ".join(f"{k}: {v.shape}" for k, v in att_patterns.items()))
            for k, v in att_patterns.items():
                run.track_artifact(v.cpu().numpy() if isinstance(v, torch.Tensor) else v, step, k)
            
        
        batch = generate_dual_task_batch(batch_size,
                                          seq_len,
                                          K,
                                          distributions,
                                          trigger_set=trigger_set)
                                        #   device=device)

        loss = evaluate_model(model,batch,loss_fn,path,device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Save sub_batch for final evaluation
    run.track_artifact(sub_batch,name='sub_batch',type='pickle')

    t1 = time.time()
    run.finalize()
    logger.info(f"Training completed in {t1-t0:.2f} seconds = {((t1-t0)/60):.2f} minutes")


if __name__ == "__main__":
    main()