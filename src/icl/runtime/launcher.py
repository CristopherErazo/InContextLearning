"""icl/launcher.py

Composition root for a single training run: builds the model/optimizer/
evaluator, wires them into a TrainerController via closures, and starts the
control-plane loop. This is the "python -m icl.launcher ..." entrypoint --
both a direct `python -m icl.launcher key=value ...` invocation and a
RunLauncher-mediated launch (from the dashboard or the `rewind` CLI) run
through this exact file.

Run-ID handshake
-----------------
When this script is launched by RunLauncher (dashboard/CLI), it's given an
`extra_args.launch_token` CLI override. RunLauncher is, at that moment,
blocked waiting to be told which run_id got claimed -- it doesn't predict
it, on purpose (see the run-id race writeup). Once ExperimentTracker.start_
run() has atomically claimed a run_id, this script calls
`rewind.launch.write_handshake(...)` to report it back. A direct
`python -m icl.launcher ...` invocation (no dashboard) simply never has a
launch_token, and this step is a no-op.

Required companion change, not made here: `icl.config.ExtraArgs` needs a
`launch_token: str | None = None` field for `cfg.extra_args.launch_token`
below to be a valid (optional) config key rather than an OmegaConf error.
"""

import sys

import torch
from omegaconf import OmegaConf

from icl import *
from tracklab import ExperimentTracker
from rewind import RunMailbox, TrainerController
from rewind.launch import write_handshake


def build_controller(cfg, log_metrics=None, log_to_terminal=None) -> TrainerController:
    """Build a fully-wired TrainerController for one run: model, optimizer,
    evaluator, and the tracker/control-plane plumbing around them.

    Parameters
    ----------
    cfg : OmegaConf
        Merged TrainerArgs (defaults + CLI overrides), already passed
        through compute_derived_args().
    log_metrics : list[str] | None
        Which eval metrics to print to the terminal log line, e.g.
        ["loss", "top1_accuracy"]. None logs every metric eval_fn returns.
    log_to_terminal : bool | None
        Whether to echo log lines to stdout in addition to the run's log
        file. None auto-detects: on only if stdout is an interactive
        terminal (so a background/dashboard-launched run stays quiet).

    Returns
    -------
    TrainerController
        Fully wired and ready for `.run_loop()`. Registering any
        project-specific command handlers (see the `perturb_layer` example
        at the bottom of this file) should happen after this returns and
        before `.run_loop()` is called.
    """
    # ---- model, loss, optimizer -----------------------------------------
    V, L = cfg.model_args.vocab_size, cfg.model_args.seq_len
    B, TB, K = cfg.data_args.batch_size, cfg.data_args.test_size, cfg.data_args.K
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MinimalTransformer(cfg.model_args).to(device)
    model.initialize_model()

    loss_fn = torch.nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer, _ = get_optimizer(trainable_params, cfg.optim_args)

    # ---- fixed evaluation batch, built once, reused every eval step ------
    test_batch = generate_icl_batch(TB, V, L, K)
    test_batch, frac_ind_possible, frac_kept = preprocess_batch(test_batch)
    evaluator = Evaluator([
        IC_TopKAccuracy(1),
        LossMetric(),
        # LossMetric(name="loss_trigg", mask="ind"),
    ])

    # ---- the three closures TrainerController actually drives ------------
    def train_step_fn():
        batch = generate_icl_batch(B, V, L, K)
        loss = evaluate_model(model, batch, loss_fn, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def eval_fn():
        return evaluator.evaluate(model, test_batch, loss_fn=loss_fn)

    def eval_art_fun():
        with torch.no_grad():
            input_ = test_batch["sequence"][:, :-1].to(device)
            mask = test_batch["mask"].to(device)
            output = model.full_output(input_, mask=mask)
        return {"logits": output["logits"].cpu()}

    # ---- tracker + run-id handshake ---------------------------------------
    exp = ExperimentTracker(cfg.extra_args.experiment_name,base_dir=cfg.extra_args.base_dir)
    tracker_run = exp.start_run(cfg, artifacts=True)

    # Tell RunLauncher (if this was launched through it) which run_id we
    # actually claimed. A direct CLI invocation never sets launch_token, so
    # this is a no-op in that case -- nothing here assumes a dashboard/CLI
    # launcher exists.
    launch_token = getattr(cfg.extra_args, "launch_token", None)
    if launch_token:
        write_handshake(exp.exp_dir, launch_token, tracker_run.run_id)

    # auto: show progress only if someone's actually watching a terminal --
    # keeps a dashboard-launched run's stdout quiet by default
    if log_to_terminal is None:
        log_to_terminal = sys.stdout.isatty()

    # ---- control plane -----------------------------------------------------
    controller = TrainerController(
        model, optimizer, cfg.extra_args.total_steps, train_step_fn, eval_fn,
        run=tracker_run,
        eval_art_fun=eval_art_fun,
        control=RunMailbox(tracker_run.run_dir) if cfg.extra_args.enable_control else None,
        enable_rewind=cfg.extra_args.enable_rewind,
        logger_kwargs={"log_to_terminal": log_to_terminal, "log_to_file": True},
        log_metrics=log_metrics or ["loss", "top1_accuracy"],
    )

    print_steps, print_steps_artifacts = get_evaluation_times(cfg.extra_args)
    controller.eval_schedule = set(print_steps)
    controller.eval_artifacts_schedule = set(print_steps_artifacts)

    controller.logger.info(f"Experiment: {cfg.extra_args.experiment_name}")
    controller.logger.info(
        f"Running {'WITH' if cfg.extra_args.enable_control else 'WITHOUT'} control features"
    )
    controller.logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    controller.logger.info(f"{frac_kept:.2%} of samples of test_batch kept after filtering")
    controller.logger.info(
        f"{frac_ind_possible:.2%} of total positions where induction is possible in test_batch"
    )

    return controller


def main():
    defaults = OmegaConf.structured(TrainerArgs())
    cli_config = OmegaConf.from_cli()
    cfg = OmegaConf.merge(defaults, cli_config)

    cfg = compute_derived_args(cfg)
    cfg.extra_args.seed, seed_msg = set_seed(cfg.extra_args.seed)

    controller = build_controller(cfg)
    controller.logger.info(seed_msg)

    controller.run_loop()


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------------- #
# Example project-specific command handler. Uncomment to try it -- it will
# show up in the dashboard's Control tab and be sendable via
# `rewind ctl perturb_layer --run <run_dir> --arg layer=<name> --arg scale=0.01`
# with zero changes needed anywhere in rewind or rewind.dashboard.
#
# IMPORTANT: passing `spec=` here isn't optional in practice -- see the
# `register_handler` bug flagged separately (a handler registered without a
# spec currently crashes run_loop() at write_actions()). Fix that in
# rewind/controller.py before relying on this example.
# ------------------------------------------------------------------------- #

# from rewind.registry import ActionSpec, ArgSpec
#
# def perturb_layer(controller, cmd):
#     """Adds Gaussian noise to one layer's weights, to probe robustness of
#     the minimum found so far."""
#     controller.snapshot_now()  # guaranteed rewind point right before the change
#     param = dict(controller.model.named_parameters())[cmd["layer"]]
#     with torch.no_grad():
#         param.add_(torch.randn_like(param) * cmd.get("scale", 0.01))
#     controller.logger.info(f"perturbed {cmd['layer']} with scale={cmd.get('scale', 0.01)}")
#
# controller.register_handler(
#     "perturb_layer", perturb_layer,
#     spec=ActionSpec(
#         "perturb_layer", "Perturb layer",
#         args={
#             "layer": ArgSpec("str", default="", description="Parameter name, e.g. 'blocks.0.attn.qkv.weight'"),
#             "scale": ArgSpec("float", default=0.01, description="Std-dev of the added noise"),
#         },
#         description="Add Gaussian noise to one layer's weights, with a snapshot taken first.",
#     ),
# )