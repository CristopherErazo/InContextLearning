from shiny import App 

from ICL.dashboards.evolution.ui import app_ui
from ICL.dashboards.evolution.server import server 


# -----------------------------
# App
# -----------------------------
app = App(app_ui, server)