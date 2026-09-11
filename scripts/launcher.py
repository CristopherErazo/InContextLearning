"""Composition root for one controllable training run (Rewind + TrackLab).

Builds model / optimizer / evaluator, wires them into a rewind.TrainerController
through three closures, opens a tracklab run and starts the loop.

    python -u scripts/launcher.py model_args.vocab_size=512 extra_args.experiment_name=my_exp
    python -m scripts.launcher ...        # same thing; this is what the dashboard spawns

Any TrainerArgs field can be overridden with OmegaConf dotted syntax. For the
same loop without Rewind (no dashboard, pause / rewind or snapshots) see
scripts/train.py.
"""
import sys

import torch
from omegaconf import OmegaConf
from rewind import RunMailbox, TrainerController
from rewind.launch import write_handshake
from tracklab import ExperimentTracker

from icl import (
    ComposedMatrices, Evaluator, LossMetric, MinimalTransformer, PerPositionOnOffLogits,
    TopKAccuracy, TrainerArgs, compute_loss, generate_icl_batch, get_evaluation_times,
    get_optimizer, load_config, preprocess_batch, set_seed,
)


def build_controller(cfg: TrainerArgs, log_metrics=None, log_to_terminal=None) -> TrainerController:
    """`log_to_terminal=None` echoes the log only when stdout is a terminal, so
    runs spawned by the dashboard or `nohup` write to the log file alone."""
    V, L = cfg.model_args.vocab_size, cfg.model_args.seq_len
    B, TB, K = cfg.data_args.batch_size, cfg.data_args.test_size, cfg.data_args.K
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- model, loss, optimizer ----
    model = MinimalTransformer(cfg.model_args).to(device)
    model.initialize_model()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer, opt_msg = get_optimizer((p for p in model.parameters() if p.requires_grad), cfg.optim_args)

    # ---- fixed test batch and probes ----
    test_batch, batch_stats = preprocess_batch(generate_icl_batch(TB, V, L, K), device)
    evaluator = Evaluator(
        scalars=[TopKAccuracy(1), LossMetric()],
        artifacts=[ComposedMatrices(), PerPositionOnOffLogits()],
        loss_fn=loss_fn,
    )

    # ---- the three closures Rewind drives ----
    # eval_fn and eval_artifacts_fn run back to back at the same step, so the
    # evaluator reuses one EvalContext (one forward pass) for both. `controller`
    # is assigned below; the closures only read it once the loop is running.
    def train_step_fn():
        loss = compute_loss(model, generate_icl_batch(B, V, L, K), loss_fn, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def eval_fn():
        return evaluator.scalars(model, test_batch, step=controller.step)

    def eval_artifacts_fn():
        return evaluator.artifacts(model, test_batch, step=controller.step)

    # ---- tracker run and launch handshake ----
    exp = ExperimentTracker(cfg.extra_args.experiment_name, cfg.extra_args.base_dir)
    run = exp.start_run(cfg, artifacts=cfg.extra_args.track_artifacts)
    # When spawned by rewind.RunLauncher (the dashboard) the child is given a
    # launch_token and must report back which run_id it claimed. A shell launch
    # never sets the token, so this is a no-op there.
    if cfg.extra_args.launch_token:
        write_handshake(exp.exp_dir, cfg.extra_args.launch_token, run.run_id)

    if log_to_terminal is None:
        log_to_terminal = sys.stdout.isatty()

    controller = TrainerController(
        model, optimizer, cfg.extra_args.total_steps, train_step_fn, run,
        eval_fn=eval_fn,
        eval_artifacts_fn=eval_artifacts_fn if cfg.extra_args.track_artifacts else None,
        control=RunMailbox(run.run_dir) if cfg.extra_args.enable_control else None,
        enable_rewind=cfg.extra_args.enable_rewind,
        logger_kwargs={"log_to_terminal": log_to_terminal, "log_to_file": True},
        log_metrics=log_metrics or ["loss", "top1_accuracy"],
    )
    eval_steps, artifact_steps = get_evaluation_times(cfg.extra_args)
    controller.eval_schedule = eval_steps
    controller.eval_artifacts_schedule = artifact_steps if cfg.extra_args.track_artifacts else set()

    log = controller.logger
    log.info(f"Experiment: {cfg.extra_args.experiment_name}")
    log.info(f"Running {'WITH' if cfg.extra_args.enable_control else 'WITHOUT'} control features, "
             f"{'WITH' if cfg.extra_args.enable_rewind else 'WITHOUT'} rewind")
    log.info(f"Device: {device}")
    log.info(opt_msg)
    log.info(batch_stats.summary())
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    return controller


def main():
    cfg = load_config()
    cfg.extra_args.seed, seed_msg = set_seed(cfg.extra_args.seed)
    controller = build_controller(cfg)
    controller.logger.info(seed_msg)
    controller.run_loop()


if __name__ == "__main__":
    main()
