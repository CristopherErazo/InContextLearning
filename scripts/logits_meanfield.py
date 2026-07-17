import time
import math
import numpy as np
from omegaconf import OmegaConf
from dataclasses import dataclass , field
import torch

from tracklab import ExperimentTracker

from icl.models.minimal_model import MinimalTransformer, initialize_model
from icl.evaluation.training import get_optimizer, evaluate_model
from icl.data.trigg_data import generate_icl_task_batch 
from icl.evaluation.utils import get_best_sub_batch, get_evaluation_times
from icl.evaluation.scalar_probes import OnOffLogitsMetric, LossMetric, IC_TopKAccuracy, EvaluatorLogits, M_Metric, Q_Metric, Gamma_Metric, Gamma_capital_Metric, Q_capital_Metric, EmpiricalLogits, Var_Metric_On, Var_Metric_Off
from icl.evaluation.tensor_probes import get_logits , get_activations
from icl.evaluation.theory import effective_loss
from icl.evaluation.utils import get_on_off_masks , get_indices


@dataclass
class ModelArgs:
    vocab_size: int = 110  # Vocabulary size
    d_model: int = 1024 # Model dimension
    seq_len: int = 32 # Sequence length
    rank: int = 50     # rank or matrices
    dropout: float = 0.0 # Dropout rate
    lin_attn: bool = True # Whether to use linear attention or not
    beta: float = 0.5 # Scaling factor for the output logits (inverse of the temperature)
    
@dataclass
class DataArgs:
    batch_size: int = 1024 # Batch size for training
    test_size: int = 1024 # Number of samples in the test set
    K : int = 8 # Number of trigger tokens  

@dataclass
class OptimArgs:
    lr: float = 0.25
    opt: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 0.0

@dataclass 
class ExtraArgs:
    total_steps: int = 5000 # Number of training steps
    n_prints: int = 50 # Number of times to print during training.
    n_prints_model: int = 10 # Number of times to save model checkpoints during training.
    print_scale: str = 'linear' # Scale for printing steps: log or linear
    experiment_name: str = 'logits' # Name of the experiment for saving results
    comments: str = '' # Additional comments for the experiment
    
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
    logger = run.get_logger(log_to_file=False, log_to_terminal=True)

    # Print the parameters
    logger.info("Experiment Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Get some parameters
    vocab_size = cfg.model_args.vocab_size
    seq_len = cfg.model_args.seq_len
    opt_name = cfg.optim_args.opt.lower()
    test_size = cfg.data_args.test_size
    batch_size = cfg.data_args.batch_size
    K = cfg.data_args.K

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Initialize Model
    model = MinimalTransformer(cfg.model_args).to(device)
    model = initialize_model(model,sigma_0=1)
    model.register_buffers(K)

    # Print trainable parameters
    logger.info("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}, {param.shape}")

    # Define loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer, message = get_optimizer(opt_name,
                                       trainable_params,
                                       cfg.optim_args.lr,
                                       cfg.optim_args.weight_decay,
                                       cfg.optim_args.momentum)
    logger.info(message)

    # Generate test batch and sub-batch for evaluation
    test_batch = generate_icl_task_batch(test_size,
                                         vocab_size,
                                         seq_len,
                                         K)
                                        #   device=device)
    
    # Get indices of the batch where is_trigg == 1 and counts > 1 (where induction can happen)
    idx_ind, perm = get_indices(test_batch,vocab_size,device=device) # shape (num_indices, 2)
    n_ind = idx_ind.shape[0]
    logger.info(f"Number of indices where induction can happen: {n_ind} out of {test_size*seq_len} total indices in the test batch.")
    logger.info(f"Percentage of indices where induction can happen: {n_ind/(test_size*seq_len)*100:.2f}%")


    # Get on/off target masks for the test batch and best sub-batch for evaluation/plotting
    on_off_masks = get_on_off_masks(test_batch, vocab_size, device=device)                              
    best_idx , sub_batch = get_best_sub_batch(test_batch, device = device, n_test=5)
    run.track_artifact(sub_batch, name="sub_batch", type="pickle")

    # Define metrics for evaluation
    metrics = [
        IC_TopKAccuracy(1),
        LossMetric(name = "loss",
                   logits_fn = lambda ctx: ctx.logits[ctx.all],
                   target_fn = lambda ctx: ctx.target[ctx.all],
                   rescale = False),       
        OnOffLogitsMetric(name = "on_target_mean",
                          mask_fn = lambda ctx: ctx.on_target_mask,
                          type = "mean"),
        OnOffLogitsMetric(name = "on_target_std",
                          mask_fn = lambda ctx: ctx.on_target_mask,
                          type = "std"),
        OnOffLogitsMetric(name = "off_target_mean",
                          mask_fn = lambda ctx: ctx.off_target_mask,
                          type = "mean"),
        OnOffLogitsMetric(name = "off_target_std",
                          mask_fn = lambda ctx: ctx.off_target_mask,
                          type = "std"),
        M_Metric(),
        Q_Metric(),
        Gamma_Metric(),
        Gamma_capital_Metric(),
        Q_capital_Metric(),
        EmpiricalLogits(),
        Var_Metric_On(),
        Var_Metric_Off()
        ]
    
    # Initialize evaluator with the defined metrics
    evaluator = EvaluatorLogits(metrics)

    # Get evaluation times based on the configuration
    total_steps = cfg.extra_args.total_steps
    print_steps, print_steps_model = get_evaluation_times(cfg.extra_args.print_scale,
                                                          cfg.extra_args.total_steps,
                                                          cfg.extra_args.n_prints,
                                                          cfg.extra_args.n_prints_model)


    t0 = time.time()

    logger.info("Starting training")
    for step in range(total_steps):
        # Evaluations and logging of scalars
        if step in print_steps:
            res = evaluator.evaluate(model, test_batch,loss_fn=loss_fn)
            theory = effective_loss(res,cfg.model_args.rank, vocab_size, K, cfg.model_args.beta,return_components=True)
            res.update(theory)

            run.track_metric(step, **res)
            logger.info(f"step {step}/{total_steps} | loss = {res['loss']:.4f} |  acc = {res['top1_accuracy']:.4f} ")
            
        # Evaluations and logging of attention patterns
        if step in print_steps_model:
            activations = get_activations(model, test_batch, 'full', device) #{'attn1': attn1, 'attn2': attn2, 'logits': logits}
            for act_name, act in activations.items():
                masks = on_off_masks[act_name]
                vmin , vmax = act.min().item(), act.max().item()
                histograms = {}
                for i , mask_name in enumerate(['on','off','all']):
                    masked_act = act[masks[mask_name]]
                    bins, edges = np.histogram(masked_act.cpu().numpy(), bins=35, range=(vmin,vmax), density=True)
                    histograms[mask_name] = bins
                histograms['edges'] = edges
                run.track_artifact(histograms, step, name=f"{act_name}_histograms",type="pickle")
                
                run.track_artifact(act[best_idx].cpu().numpy(), step, name=f"{act_name}_matrix",type="tensor")
            


            logits = activations['logits']
            logits_ind = logits[idx_ind[:,0], idx_ind[:,1],:] # shape (num_indices, vocab_size)
            permuted_logits = logits_ind.gather(1, perm) # shape (num_indices, vocab_size)
            run.track_artifact(permuted_logits.cpu().numpy(), step, name=f"logits_ind",type="tensor")

            # # Compute effective variable moments
            # h_star = permuted_logits[:,-1] # shape (num_indices,)
            # logS = torch.logsumexp(permuted_logits[:,:-1], dim=1) # shape (num_indices,)

            # h_star_mean = h_star.mean().item()
            # logS_mean = logS.mean().item()
            # h_star_std = h_star.std().item()
            # logS_std = logS.std().item()
            
            # h_star_logS_corr = (h_star - h_star_mean)*(logS - logS_mean) # shape (num_indices,)
            # h_star_logS_corr = h_star_logS_corr.mean().item()


            # run.track_metric(step, h_star_mean = h_star_mean,
            #                         logS_mean = logS_mean,
            #                         h_star_std = h_star_std,
            #                         logS_std = logS_std,
            #                         h_star_logS_corr = h_star_logS_corr)            
            

            

        batch = generate_icl_task_batch(batch_size,
                                        vocab_size,
                                        seq_len,
                                        K)
                                        #   device=device)

        loss = evaluate_model(model,batch,loss_fn,'full',device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

    t1 = time.time()
    run.finalize()
    logger.info(f"Training Run: {run.run_id} completed in {t1-t0:.2f} seconds = {((t1-t0)/60):.2f} minutes")


if __name__ == "__main__":
    main()