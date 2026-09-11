"""
Entry point launched as its own OS process by the dashboard:

    python -m icl.runtime.service extra_args.experiment_name=run1 model_args.d_model=64

The build itself (model, optimizer, data, controller) lives in
icl.runtime.build -- shared with plain main.py. This file only adds what's
specific to being launched as a controllable, dashboard-driven run.
"""

import torch
from omegaconf import OmegaConf

from icl.utils import set_seed
from icl.config.default_config import TrainerArgs
from icl.runtime.build import build_controller


def perturb_layer(controller, cmd):
    """Example project-specific handler: adds Gaussian noise to one layer's
    weights, to probe robustness of the minimum found so far."""
    controller.snapshot_now()  # guaranteed rewind point right before the change
    param = dict(controller.model.named_parameters())[cmd["layer"]]
    with torch.no_grad():
        param.add_(torch.randn_like(param) * cmd.get("scale", 0.01))
    controller.logger.info(f"perturbed {cmd['layer']} with scale={cmd.get('scale', 0.01)}")


if __name__ == "__main__":
    defaults = OmegaConf.structured(TrainerArgs())
    cli_config = OmegaConf.from_cli()
    cfg = OmegaConf.merge(defaults, cli_config)
    cfg.extra_args.seed, _ = set_seed(cfg.extra_args.seed)

    controller = build_controller(cfg)
    controller.register_handler("perturb_layer", perturb_layer)
    controller.run_loop()