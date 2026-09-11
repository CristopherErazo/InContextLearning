"""
Builds everything one training run needs -- model, optimizer, data,
evaluator -- and wires it into a TrainerController pointed at a tracklab
run. Used by both main.py (a plain run, no live control) and
runtime/service.py (a run launched by the dashboard), so the loop,
logging, and tracking logic is written and tested in exactly one place.
"""

import sys

import torch

from tracklab import ExperimentTracker
from icl.models.minimal_full_rank import MinimalTransformer
from icl.evaluation.training import evaluate_minimal_model, get_optimizer
from icl.evaluation.scalar_measures import Evaluator, LossMetric, IC_TopKAccuracy
from icl.data.trigg_data import generate_icl_task_batch
from icl.evaluation.utils import preprocess_batch 

from rewind.tracker_adapter import ControllableRun
from rewind.controller import TrainerController






def build_controller(cfg, log_metrics=None, log_to_terminal=None) -> TrainerController:
    V, L = cfg.model_args.vocab_size, cfg.model_args.seq_len
    B, TB, K = cfg.data_args.batch_size, cfg.data_args.test_size, cfg.data_args.K
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MinimalTransformer(cfg.model_args).to(device)
    model.initialize_model()

    loss_fn = torch.nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer, _ = get_optimizer(trainable_params, cfg.optim_args)

    test_batch = generate_icl_task_batch(TB, V, L, K)
    test_batch, frac_ind_possible, frac_kept = preprocess_batch(test_batch)
    evaluator = Evaluator([
        IC_TopKAccuracy(1),
        LossMetric(),
        # LossMetric(name="loss_trigg", mask="ind"),
    ])

    def train_step_fn():
        batch = generate_icl_task_batch(B, V, L, K)
        loss = evaluate_minimal_model(model, batch, loss_fn, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def eval_fn():
        return evaluator.evaluate(model, test_batch, loss_fn=loss_fn)

    exp = ExperimentTracker(cfg.extra_args.experiment_name)
    tracklab_run = exp.start_run(cfg, artifacts=True)
    run = ControllableRun(tracklab_run)

    if log_to_terminal is None:
        log_to_terminal = sys.stdout.isatty()  # auto: show progress only if someone's watching

    controller = TrainerController(
        model, optimizer, train_step_fn, eval_fn, cfg.extra_args.total_steps, run,
        logger_kwargs={"log_to_terminal": log_to_terminal, "log_to_file": True},
        log_metrics=log_metrics or ["loss", "top1_accuracy"],
    )

    n_prints = cfg.extra_args.n_prints
    step_gap = max(1, cfg.extra_args.total_steps // n_prints)
    controller.eval_schedule = set(range(0, cfg.extra_args.total_steps, step_gap))

    return controller