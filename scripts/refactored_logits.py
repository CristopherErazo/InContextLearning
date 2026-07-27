import torch
import time
from omegaconf import OmegaConf

from tracklab import ExperimentTracker


from icl.models.minimal_full_rank import MinimalTransformer
from icl.evaluation.training import get_optimizer, evaluate_minimal_model
from icl.evaluation.scalar_measures import Evaluator, LossMetric, IC_TopKAccuracy, AvgAccuracy, LogitStatistics
from icl.evaluation.utils import preprocess_batch 
from icl.data.trigg_data import generate_icl_task_batch 
from icl.evaluation.utils import get_evaluation_times
from icl.utils import set_seed

# Import configuration components from scripts/config.py
from default_config import TrainerArgs, compute_derived_args


def main():
    # Load configuration from command line and merge with defaults
    defaults = OmegaConf.structured(TrainerArgs())
    cli_config = OmegaConf.from_cli()
    cfg = OmegaConf.merge(defaults, cli_config)

    # Recompute dynamic fields using merged CLI overrides
    cfg = compute_derived_args(cfg)

    # Set random seed for reproducibility
    cfg.extra_args.seed, msg = set_seed(cfg.extra_args.seed)

    # Create an experiment and run
    experiment_name = cfg.extra_args.experiment_name
    exp = ExperimentTracker(experiment_name)
    run  = exp.start_run(cfg, artifacts=True)

    # Initialize logger for the run
    logger = run.get_logger(log_to_file=False, log_to_terminal=True)

    # Print the parameters
    logger.info("Experiment Configuration:")
    logger.info(msg)

    # Get some parameters
    V = cfg.model_args.vocab_size
    L = cfg.model_args.seq_len
    TB = cfg.data_args.test_size
    B = cfg.data_args.batch_size
    K = cfg.data_args.K

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Initialize Model
    model = MinimalTransformer(cfg.model_args).to(device)
    model.initialize_model()    

    # Print trainable parameters
    logger.info("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}, {param.shape}")

    # # Define loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer, msg = get_optimizer(trainable_params,cfg.optim_args)
    logger.info(msg)

    # Generate test batch and sub-batch for evaluation
    test_batch = generate_icl_task_batch(TB,V,L,K) 
    test_batch, frac_ind_possible, frac_kept = preprocess_batch(test_batch)

    logger.info(f"Fraction of sequences kept after preprocessing: {frac_kept:.4f}")
    logger.info(f"Fraction of positions where induction is possible: {frac_ind_possible:.4f}")
    logger.info(f"New test batch size: {test_batch['sequence'].shape[0]} ")
 
    # Define metrics for evaluation
    metrics = [
        LossMetric(),       
        LossMetric(name = "loss_ind",mask = 'ind'),
        LossMetric(name = "loss_no_ind",mask = 'no_ind'),
        IC_TopKAccuracy(1),
        AvgAccuracy(),
        LogitStatistics()             
        ]
    
    # Initialize evaluator with the defined metrics
    evaluator = Evaluator(metrics)

    # Get evaluation times based on the configuration
    total_steps = cfg.extra_args.total_steps
    print_steps, print_steps_model = get_evaluation_times(cfg.extra_args)

    t0 = time.time()

    logger.info("Starting training")
    for step in range(total_steps):
        # Evaluations and logging of scalars
        if step in print_steps:
            # Evaluate the model on the test batch and log the results
            res = evaluator.evaluate(model, test_batch,loss_fn=loss_fn)
            run.track_metric(step, **res)
            logger.info(f"step {step}/{total_steps} | loss = {res['loss_ind']:.2f} |  acc = {res['top1_accuracy']:.2f} | avg_acc = {res['avg_accuracy']:.2f}")


        # Training step: Generate a new batch, compute loss, backpropagate, and update model parameters
        batch = generate_icl_task_batch(B,V,L,K) 
        loss = evaluate_minimal_model(model,batch,loss_fn,device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()        

    t1 = time.time()
    run.finalize()
    logger.info(f"Training Run: {run.run_id} completed in {t1-t0:.2f} seconds = {((t1-t0)/60):.2f} minutes")


if __name__ == "__main__":
    main()