"""Rewind dashboard for this project.

Everything generic (launch form, status, live metrics plot, pause / resume /
set-lr / rewind controls) comes from rewind.dashboard. This file only says
which config class to build the launch form from, where runs live, and which
module to spawn for a new run.

Run from the repo root (the entrypoint is imported as `python -m scripts.launcher`):

    shiny run --reload scripts/dash.py
"""

from rewind.dashboard import DashboardConfig, build_dashboard

from icl import TrainerArgs

app = build_dashboard(
    DashboardConfig(
        base_dir="./data",
        config_cls=TrainerArgs,
        entrypoint="scripts.launcher",
        poll_interval_s=1.0,
    )
)
