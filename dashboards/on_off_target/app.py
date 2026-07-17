from shiny import App 

# from ICL.dashboards.on_off_target.ui import app_ui
# from ICL.dashboards.on_off_target.server import server 
# from ICL.dashboards.on_off_target.server import server

from server import server
from ui import app_ui


# -----------------------------
# App
# -----------------------------
app = App(app_ui, server)