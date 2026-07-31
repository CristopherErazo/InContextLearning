"""
This project's dashboard. Everything generic (launch form, status, plot,
pause/resume/LR/rewind controls) comes from rewind.dashboard -- this file
only supplies what's specific to icl: which config to build the launch
form from, and which service module to launch.
"""

from rewind.dashboard import build_dashboard
from icl.config.default_config import TrainerArgs

app = build_dashboard(
    config_cls=TrainerArgs,
    service_module="icl.runtime.service",
    experiment_name="logits",
)