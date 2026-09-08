import sys
import time
import torch
from omegaconf import OmegaConf
import numpy as np

from icl import *
from tracklab import ExperimentTracker
from rewind import TrainerController, RunMailbox


def build_controller(cfg : TrainerArgs, log_metrics=None, log_to_terminal=True) -> TrainerController:

    # Set up model, loss function, optimizer
    V, L = cfg.model_args.vocab_size, cfg.model_args.seq_len
    B, TB, K = cfg.data_args.batch_size, cfg.data_args.test_size, cfg.data_args.K
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MinimalTransformer(cfg.model_args).to(device)
    model.initialize_model()

    loss_fn = torch.nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer, _ = get_optimizer(trainable_params, cfg.optim_args)


    # Generate a test batch and preprocess it for evaluation
    test_batch = generate_icl_batch(TB, V, L, K)
    test_batch, frac_ind_possible, frac_kept = preprocess_batch(test_batch)
    evaluator = Evaluator([IC_TopKAccuracy(1),
                           LossMetric(),
                        #    ExpectedOnTargetLogit(),
                        #    LogitStatistics(),
                           # LossMetric(name="loss_trigg", mask="ind"),
                           ])

    # Define the training step function and evaluation functions
    def train_step_fn():
        batch = generate_icl_batch(B, V, L, K)
        loss = evaluate_model(model, batch, loss_fn, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # time.sleep(.2)  # simulate a long step so we can see the dashboard update

    def eval_fn():
        return evaluator.evaluate(model, test_batch, loss_fn=loss_fn)

    def eval_art_fun():
        # hst = get_logit_distributions(model, test_batch, device,n_bins=100)
        mtx = model.get_composed_matrices()
        # means = get_per_position_on_logit_mean(model, test_batch, device)
        on_off_logits = get_per_position_on_off_logits(model, test_batch, device)
        mtx.update({("hists","logits") : (on_off_logits,'pickle')})
        return mtx


    # Set up the experiment tracker and the TrainerController
    exp = ExperimentTracker(cfg.extra_args.experiment_name, cfg.extra_args.base_dir)
    tracker_run = exp.start_run(cfg, artifacts=cfg.extra_args.track_artifacts)

    log_to_terminal = sys.stdout.isatty() if log_to_terminal is None else log_to_terminal # auto: show progress only if someone's watching

    controller = TrainerController(
        model, optimizer, cfg.extra_args.total_steps, train_step_fn, eval_fn, 
        run = tracker_run,
        eval_art_fun = eval_art_fun if cfg.extra_args.track_artifacts else None,
        control = RunMailbox(tracker_run.run_dir) if cfg.extra_args.enable_control else None,
        enable_rewind = cfg.extra_args.enable_rewind,
        logger_kwargs={"log_to_terminal": log_to_terminal, "log_to_file": True},
        log_metrics=log_metrics or ["loss", "top1_accuracy"])

    # Set the evaluation schedule based on the configuration
    print_steps, print_steps_artifacts = get_evaluation_times(cfg.extra_args)

    #custom_steps 
    custom_steps = np.array([*np.linspace(0,360,5),*np.linspace(361,500,20),*np.linspace(501,800,5)]).astype(int).tolist()

    controller.eval_schedule = set(print_steps)  # override eval_schedule with custom steps
    controller.eval_artifacts_schedule = set(print_steps_artifacts) if cfg.extra_args.track_artifacts else set()
    # controller.total_steps = custom_steps[-1]  # override total_steps with the last custom step
    # Log initial messages about the experiment and configuration
    controller.logger.info(f"Experiment: {cfg.extra_args.experiment_name}")
    controller.logger.info(f"Running {['WITHOUT', 'WITH'][cfg.extra_args.enable_control]} control features")
    controller.logger.info(f"Running in device: {device}")
    controller.logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    controller.logger.info(f"{frac_kept:.2%} of samples of test_batch kept after filtering")
    controller.logger.info(f"{frac_ind_possible:.2%} of total positions where induction is possible in test_batch")

    return controller



def main():
    # Load configuration from command line arguments and merge with defaults
    defaults = OmegaConf.structured(TrainerArgs())
    cli_config = OmegaConf.from_cli()
    cfg = OmegaConf.merge(defaults, cli_config)

    # Compute derived arguments and set random seed
    cfg = compute_derived_args(cfg)
    cfg.extra_args.seed, msg = set_seed(cfg.extra_args.seed)

    # Build the TrainerController
    controller = build_controller(cfg,log_to_terminal=False)
    controller.logger.info(msg)
    
    controller.run_loop()


if __name__ == "__main__":
    main()


# def perturb_layer(controller, cmd):
#     """Example project-specific handler: adds Gaussian noise to one layer's
#     weights, to probe robustness of the minimum found so far."""
#     controller.snapshot_now()  # guaranteed rewind point right before the change
#     param = dict(controller.model.named_parameters())[cmd["layer"]]
#     with torch.no_grad():
#         param.add_(torch.randn_like(param) * cmd.get("scale", 0.01))
#     controller.logger.info(f"perturbed {cmd['layer']} with scale={cmd.get('scale', 0.01)}")

#controller.register_handler("perturb_layer", perturb_layer)