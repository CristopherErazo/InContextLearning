"""icl/scripts/dashboard_advanced.py

The stress-test dashboard: exercises every extension point the engine
offers, in one script, so a change to rewind.dashboard's core can be
checked against all of them at once rather than one at a time.

Covers:
  1. A field_override -- optim_args.opt_name rendered as a constrained
     dropdown instead of the default free-text box a bare `str` field gets.
  2. A DashboardExtension -- a "Raw config" tab that isn't part of the
     baseline engine at all, added purely by this project.
  3. (Documented, not implemented here -- see the bottom of this file) the
     training-side half of a custom, "weird" control-plane action with
     arguments, to exercise control_panel.py's "form" rendering path beyond
     the built-in set_lr example.

Run with:
    shiny run --reload icl/scripts/dashboard_advanced.py
"""

from __future__ import annotations

from pathlib import Path

from shiny import render, ui

from rewind.dashboard import DashboardConfig, RunContext, build_dashboard
from rewind.dashboard.schema import FieldSpec, input_id

from icl.config import TrainerArgs

# --------------------------------------------------------------------- #
# 1. field_override: optim_args.opt_name is a plain `str` in TrainerArgs,
#    so the default widget would be a free-text box -- anyone could type
#    "adamm" and not find out until the subprocess fails to construct an
#    optimizer. A project-supplied override fixes that for this one field
#    without touching how every other field renders.
# --------------------------------------------------------------------- #

_OPTIMIZER_CHOICES = ["adam", "sgd", "adamw"]


def _optimizer_widget(f: FieldSpec):
    return ui.input_select(
        input_id(f.path),  # MUST match this convention -- see schema.input_id's docstring
        "Optimizer",
        choices=_OPTIMIZER_CHOICES,
        selected=f.default if f.default in _OPTIMIZER_CHOICES else _OPTIMIZER_CHOICES[0],
    )


# --------------------------------------------------------------------- #
# 2. DashboardExtension: a tab the baseline engine doesn't provide at all.
#    Deliberately simple -- pretty-prints the active run's config.json --
#    so it's easy to tell "does the extension mechanism work" apart from
#    "is this particular extension's content correct".
# --------------------------------------------------------------------- #


class RawConfigExtension:
    id = "raw_config"
    label = "Raw config"

    def ui(self):
        return ui.output_text_verbatim("raw_config_text")

    def server(self, input, output, session, ctx: RunContext):
        @render.text
        def raw_config_text():
            # ctx.run_dir is a callable (see RunContext's docstring) --
            # calling it here, inside a @render function, is what keeps
            # this reactive to run selection instead of freezing at
            # whatever was active when the app started.
            run_dir = ctx.run_dir()
            if run_dir is None:
                return "No active run selected."
            config_path = Path(run_dir) / "config.json"
            if not config_path.exists():
                return "config.json not written yet."
            return config_path.read_text()


app = build_dashboard(
    DashboardConfig(
        base_dir="./data",
        config_cls=TrainerArgs,
        entrypoint="icl.launcher",
        field_overrides={"optim_args.opt_name": _optimizer_widget},
        extensions=[RawConfigExtension()],
    )
)

# --------------------------------------------------------------------- #
# 3. The companion piece this script CANNOT provide on its own: a custom
#    control-plane action with arguments, to see control_panel.py render a
#    "form" (not just a bare button) for something other than the built-in
#    set_lr. That half lives in the training process, not the dashboard --
#    add something like this to icl/launcher.py's build_controller(), and
#    it will appear in the Control tab automatically, with zero changes
#    here:
#
#     from rewind.registry import ActionSpec, ArgSpec
#
#     def _inject_noise(cmd, *, model, **_):
#         std = float(cmd.get("std", 0.01))
#         with torch.no_grad():
#             for p in model.parameters():
#                 p.add_(torch.randn_like(p) * std)
#
#     controller.register_handler(
#         "inject_noise",
#         lambda cmd: _inject_noise(cmd, model=model),
#         spec=ActionSpec(
#             "inject_noise", "Inject noise",
#             args={"std": ArgSpec("float", default=0.01,
#                                   description="Std-dev of noise added to every parameter")},
#             description="Nudge all weights with iid Gaussian noise -- for probing loss landscape sensitivity.",
#         ),
#     )
# --------------------------------------------------------------------- #