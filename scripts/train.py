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
learning time. `extra_args.stop_at_loss` (default None) does the same when the
loss falls to the threshold (e.g. a fraction of the way to `asymptotic_loss`). Its resolution is the scalar evaluation spacing, `total_steps/n_prints`.
A non-finite evaluation loss stops the run unconditionally: the learning rate has
blown the weights up and nothing after that point is meaningful. Both tests read the
metrics already computed by the scheduled evaluation, so neither adds a device sync
to the training step.

`model_args.backend=reduced` trains the same model in the (M, Q, G) coordinates
(icl.reduced): exact for SGD, and a step costs the same at any d_model. With
`model_args.init=sample` it also runs at `model_args.infinite_d=true`.
"""
import math
import sys
import time

import torch
from omegaconf import OmegaConf
from tracklab import ExperimentTracker

from icl import (
    ComposedMatrices, Evaluator, LossMetric, OrderParameters, TopKAccuracy, TrainerArgs,
    TriggerLogitTable, build_model,
    compute_loss, generate_icl_batch, get_evaluation_times, load_config, log_artifacts,
    preprocess_batch, set_matmul_precision, set_seed,
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
    stop_at_loss = cfg.extra_args.stop_at_loss
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Where batches are drawn. "auto" follows the training device, which avoids the
    # host->device copy; see scripts/bench_data.py for which is faster on this machine.
    gen_device = device if cfg.data_args.gen_device == "auto" else cfg.data_args.gen_device
    # Set before any tensor is touched: it decides whether float32 matmuls run on
    # the tensor cores (TF32) or the fp32 units, which changes numerics, not just
    # speed. Recorded in the saved config so runs stay comparable.
    precision_msg = set_matmul_precision(cfg.extra_args.matmul_precision)

    # ---- model, loss, optimizer ----
    model, optimizer, opt_msg = build_model(cfg.model_args, cfg.optim_args, device, K=K)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ---- fixed test batch, probes, schedules ----
    test_batch, batch_stats = preprocess_batch(generate_icl_batch(TB, V, L, K, device=gen_device), device)
    # The test batch is evaluated `eval_chunk` sequences at a time. Its default, the
    # training batch size, needs less memory than a training step (no autograd graph).
    evaluator = Evaluator(
        scalars=[TopKAccuracy(1),
                 LossMetric(),
                 OrderParameters()],
        artifacts=[ComposedMatrices(),
                   TriggerLogitTable(cfg.extra_args.logit_positions)],
        chunk=cfg.extra_args.eval_chunk or B,
    )
    eval_steps, artifact_steps = get_evaluation_times(cfg.extra_args)
    if not track_artifacts:
        artifact_steps = set()
    log_metrics = log_metrics or ["loss", "top1_accuracy"]

    eval_time = 0.0  # seconds spent in evaluation, reported apart from training time

    def save_artifacts(run, log, step: int) -> None:
        artifacts = evaluator.artifacts(model, test_batch, step=step)
        log_artifacts(run, artifacts, step)
        log.info(f"saved {len(artifacts)} eval artifact(s) at step {step}")

    def evaluate(run, log, step: int, artifacts_off_schedule: bool = False) -> dict[str, float] | None:
        """Scalars and artifacts for `step`; both read one shared EvalContext.

        Returns the scalar metrics when this step is on the scalar schedule, so the
        loop can test the early-stopping criterion without a second forward pass.
        `artifacts_off_schedule` saves artifacts at a step the schedule skips (an
        early stop's final weights) and computes nothing else.
        """
        nonlocal eval_time
        if device == "cuda":
            torch.cuda.synchronize()  # don't bill queued training kernels to evaluation
        t = time.perf_counter()
        metrics = None
        if artifacts_off_schedule:
            if track_artifacts and step not in artifact_steps:
                save_artifacts(run, log, step)
        else:
            if step in eval_steps:
                metrics = evaluator.scalars(model, test_batch, step=step)
                run.track_metric(step, **metrics)
                log.info(format_metrics(step, total_steps, metrics, log_metrics))
            if step in artifact_steps:
                save_artifacts(run, log, step)
        eval_time += time.perf_counter() - t
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
        log.info(precision_msg)
        log.info(batch_stats.summary())
        log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        step, stopped, t0 = 0, False, time.perf_counter()
        try:
            for step in range(total_steps):
                metrics = evaluate(run, log, step)
                # A non-finite loss means the run is dead (too large optim_args.alpha_lr);
                # nothing after this point is meaningful, so stop instead of burning the budget.
                if metrics is not None and not math.isfinite(metrics.get("loss", 0.0)):
                    log.error(f"diverged at step {step}: loss={metrics['loss']} -- lower optim_args.alpha_lr")
                    stopped = True
                    break
                reached_accuracy = stop_at is not None and metrics is not None and metrics.get("top1_accuracy", 0.0) >= stop_at
                reached_loss = stop_at_loss is not None and metrics is not None and metrics.get("loss", math.inf) <= stop_at_loss
                if reached_accuracy or reached_loss:
                    log.info(f"early stop at step {step}: top1_accuracy={metrics['top1_accuracy']:.4f}, "
                             f"loss={metrics['loss']:.4f} (stop_at_accuracy={stop_at}, stop_at_loss={stop_at_loss})")
                    stopped = True
                    evaluate(run, log, step, artifacts_off_schedule=True)  # final weights
                    break
                loss = compute_loss(model, generate_icl_batch(B, V, L, K, stats=False, device=gen_device),
                            loss_fn, device)
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
        finally:
            # Last line of the log whatever happened: normal end, early stop, divergence,
            # KeyboardInterrupt or crash. `finally` runs before the exception propagates.
            elapsed = time.perf_counter() - t0
            train_time = elapsed - eval_time
            ms_per_step = 1000 * train_time / max(step, 1)
            peak_gib = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
            log.info(
                f"elapsed time = ({elapsed / 60:.2f} min) = ({elapsed/3600:.2f} hr) for {step} steps: "
                f"train {train_time / 60:.2f} min ({ms_per_step:.1f} ms/step), eval {eval_time / 60:.2f} min | "
                f"peak GPU memory {peak_gib:.2f} GiB"
            )
            run.track_results(
                elapsed_time=elapsed,
                train_time=train_time,
                eval_time=eval_time,
                steps_completed=step,
                ms_per_step=ms_per_step,
                peak_gpu_mem_gib=peak_gib)

def main():
    cfg = load_config()
    cfg.extra_args.seed, seed_msg = set_seed(cfg.extra_args.seed)
    print(seed_msg)
    train(cfg,log_to_terminal=cfg.extra_args.log_to_terminal)


if __name__ == "__main__":
    main()
