"""Plain training run: the same model, loss, evaluation schedule and artifacts
as scripts/launcher.py, but the loop is written out here and only TrackLab is
used. No dashboard, no pause / resume / rewind.

    python -u scripts/train.py model_args.vocab_size=512 extra_args.experiment_name=my_exp

Any TrainerArgs field can be overridden with OmegaConf dotted syntax. The
`extra_args.enable_control`, `enable_rewind` and `launch_token` fields are
ignored here; `track_artifacts` and the `n_prints*` / `print_scale` schedule
fields behave exactly as in the launcher.

Loop convention (shared with rewind.TrainerController): evaluation at step `s`
sees the weights *before* the s-th update, so step 0 is the initial model and
`total_steps` is the final one.
"""
import sys

import torch
from omegaconf import OmegaConf
from tracklab import ExperimentTracker

from icl import (
    ComposedMatrices, Evaluator, LossMetric, MinimalTransformer, PerPositionOnOffLogits,
    TopKAccuracy, TrainerArgs, compute_loss, generate_icl_batch, get_evaluation_times,
    get_optimizer, load_config, log_artifacts, preprocess_batch, set_seed,
)


def format_metrics(step: int, total: int, metrics: dict[str, float], keys=None) -> str:
    keys = keys if keys is not None else list(metrics)
    parts = [f"{k}={metrics[k]:.4f}" for k in keys if k in metrics]
    return f"step {step}/{total} | " + " | ".join(parts)


def train(cfg: TrainerArgs, log_metrics=None, log_to_terminal=None) -> None:
    V, L = cfg.model_args.vocab_size, cfg.model_args.seq_len
    B, TB, K = cfg.data_args.batch_size, cfg.data_args.test_size, cfg.data_args.K
    total_steps = cfg.extra_args.total_steps
    track_artifacts = cfg.extra_args.track_artifacts
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- model, loss, optimizer ----
    model = MinimalTransformer(cfg.model_args).to(device)
    model.initialize_model()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer, opt_msg = get_optimizer((p for p in model.parameters() if p.requires_grad), cfg.optim_args)

    # ---- fixed test batch, probes, schedules ----
    test_batch, batch_stats = preprocess_batch(generate_icl_batch(TB, V, L, K), device)
    evaluator = Evaluator(
        scalars=[TopKAccuracy(1), LossMetric()],
        artifacts=[ComposedMatrices(), PerPositionOnOffLogits()],
        loss_fn=loss_fn,
    )
    eval_steps, artifact_steps = get_evaluation_times(cfg.extra_args)
    if not track_artifacts:
        artifact_steps = set()
    log_metrics = log_metrics or ["loss", "top1_accuracy"]

    def evaluate(run, log, step: int) -> None:
        """Scalars and artifacts for `step`; both read one shared EvalContext."""
        if step in eval_steps:
            metrics = evaluator.scalars(model, test_batch, step=step)
            run.track_metric(step, **metrics)
            log.info(format_metrics(step, total_steps, metrics, log_metrics))
        if step in artifact_steps:
            artifacts = evaluator.artifacts(model, test_batch, step=step)
            log_artifacts(run, artifacts, step)
            log.info(f"saved {len(artifacts)} eval artifact(s) at step {step}")

    # ---- the run ----
    exp = ExperimentTracker(cfg.extra_args.experiment_name, cfg.extra_args.base_dir)
    with exp.start_run(cfg, artifacts=track_artifacts) as run:
        if log_to_terminal is None:
            log_to_terminal = sys.stdout.isatty()
        log = run.get_logger(log_to_terminal=log_to_terminal, log_to_file=True)
        log.info(f"Experiment: {cfg.extra_args.experiment_name} | run_id={run.run_id} | total_steps={total_steps}")
        log.info(f"Device: {device}")
        log.info(opt_msg)
        log.info(batch_stats.summary())
        log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

        step = 0
        try:
            for step in range(total_steps):
                evaluate(run, log, step)
                loss = compute_loss(model, generate_icl_batch(B, V, L, K), loss_fn, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            step = total_steps
            evaluate(run, log, step)  # final weights, if the schedule asks for them
        except KeyboardInterrupt:
            log.warning(f"interrupted at step {step}")
            raise
        except Exception:
            log.exception(f"training crashed at step {step}")
            raise
        else:
            log.info(f"training done at step {step}")


def main():
    cfg = load_config()
    cfg.extra_args.seed, seed_msg = set_seed(cfg.extra_args.seed)
    print(seed_msg)
    train(cfg,log_to_terminal=True)


if __name__ == "__main__":
    main()
