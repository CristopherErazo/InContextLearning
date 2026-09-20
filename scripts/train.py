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

`extra_args.stop_at_accuracy` (default None) stops the run at the first scheduled
evaluation whose `top1_accuracy` reaches the threshold; that step is the measured
learning time. Its resolution is the scalar evaluation spacing, `total_steps/n_prints`.
A non-finite evaluation loss stops the run unconditionally: the learning rate has
blown the weights up and nothing after that point is meaningful. Both tests read the
metrics already computed by the scheduled evaluation, so neither adds a device sync
to the training step.
"""
import math
import sys

import torch
from omegaconf import OmegaConf
from tracklab import ExperimentTracker

from icl import (
    ComposedMatrices, Evaluator, LossMetric, MinimalTransformer, PerPositionOnOffLogits,
    TopKAccuracy, TrainerArgs, compute_loss, generate_icl_batch, get_evaluation_times,
    get_optimizer, load_config, log_artifacts, preprocess_batch, set_seed, AttentionMaps
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
    stop_at = cfg.extra_args.stop_at_accuracy  # None disables early stopping
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
        artifacts=[AttentionMaps()],
        loss_fn=loss_fn,
    )
    eval_steps, artifact_steps = get_evaluation_times(cfg.extra_args)
    if not track_artifacts:
        artifact_steps = set()
    log_metrics = log_metrics or ["loss", "top1_accuracy"]

    def evaluate(run, log, step: int) -> dict[str, float] | None:
        """Scalars and artifacts for `step`; both read one shared EvalContext.

        Returns the scalar metrics when this step is on the scalar schedule, so the
        loop can test the early-stopping criterion without a second forward pass.
        """
        metrics = None
        if step in eval_steps:
            metrics = evaluator.scalars(model, test_batch, step=step)
            run.track_metric(step, **metrics)
            log.info(format_metrics(step, total_steps, metrics, log_metrics))
        if step in artifact_steps:
            artifacts = evaluator.artifacts(model, test_batch, step=step)
            log_artifacts(run, artifacts, step)
            log.info(f"saved {len(artifacts)} eval artifact(s) at step {step}")
        return metrics

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

        step, stopped = 0, False
        try:
            for step in range(total_steps):
                metrics = evaluate(run, log, step)
                # A non-finite loss means the run is dead (too large optim_args.alpha_lr);
                # nothing after this point is meaningful, so stop instead of burning the budget.
                if metrics is not None and not math.isfinite(metrics.get("loss", 0.0)):
                    log.error(f"diverged at step {step}: loss={metrics['loss']} -- lower optim_args.alpha_lr")
                    stopped = True
                    break
                if stop_at is not None and metrics is not None and metrics.get("top1_accuracy", 0.0) >= stop_at:
                    log.info(f"early stop at step {step}: top1_accuracy={metrics['top1_accuracy']:.4f} >= {stop_at}")
                    stopped = True
                    break
                loss = compute_loss(model, generate_icl_batch(B, V, L, K), loss_fn, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if not stopped:
                step = total_steps
                evaluate(run, log, step)  # final weights, if the schedule asks for them
        except KeyboardInterrupt:
            log.warning(f"interrupted at step {step}")
            raise
        except Exception:
            log.exception(f"training crashed at step {step}")
            raise
        else:
            log.info(f"training {'stopped' if stopped else 'done'} at step {step}")


def main():
    cfg = load_config()
    cfg.extra_args.seed, seed_msg = set_seed(cfg.extra_args.seed)
    print(seed_msg)
    train(cfg,log_to_terminal=True)


if __name__ == "__main__":
    main()
