from shiny import App 

from ui import app_ui
from server import server 


# -----------------------------
# App
# -----------------------------
app = App(app_ui, server)