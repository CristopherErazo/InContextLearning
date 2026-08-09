"""icl/scripts/dashboard_basic.py

The simplest possible Rewind dashboard for this project: no custom widgets,
no custom control-plane actions, no extension tabs. Every panel here is the
engine's baseline behavior, driven entirely by introspecting TrainerArgs.

This is also the reference sanity check for the engine itself: if this
script ever breaks, the problem is in rewind.dashboard, not in anything
project-specific -- there is nothing project-specific here to have broken.

Run with:
    shiny run --reload icl/scripts/dashboard_basic.py
"""

from rewind.dashboard import DashboardConfig, build_dashboard

from icl.config import TrainerArgs

app = build_dashboard(
    DashboardConfig(
        base_dir="./data",
        config_cls=TrainerArgs,
        entrypoint="icl.runtime.launcher",
    )
)