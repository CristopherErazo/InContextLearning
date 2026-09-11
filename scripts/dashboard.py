import os
import sys
import subprocess
import time
import pandas as pd
from pathlib import Path
from shiny import App, ui, render, reactive

from icl import TrainerArgs
from tracklab import ExperimentReader, ExperimentTracker

NEW_EXP_MSG = "Create New Experiment!"
DEFAULT_MSG = "-- Select an experiment --"

# Get all the names of folders in the ./data directory, most recently
# modified first.
exp_names = [f.name for f in os.scandir("./data") if f.is_dir()]
exp_names.sort(key=lambda f: os.path.getmtime(os.path.join("./data", f)), reverse=True)
exp_names = [DEFAULT_MSG, NEW_EXP_MSG, *exp_names]


app_ui = ui.page_fluid(
        # MathJax for LaTeX rendering
        # ui.tags.head(ui.tags.script(src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")),
        ui.h1("Training Hypervisor"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h3("Select or Create Experiment"),
                ui.input_select("exp_name", None, choices=exp_names),
                ui.panel_conditional(
                    f"input.exp_name == '{NEW_EXP_MSG}'",
                    ui.input_submit_textarea("new_exp_name", None, placeholder="New Experiment Name"),
                ),
                ui.output_ui("launch_or_inspect_checkbox"),
                ui.output_ui("runs_inspector"),
            ),
            ui.output_text("tmp"),
            ui.output_ui("runs_launcher"),
            ui.output_ui("live_dash")
        ),
    )


def server(input, output, session):
    # ---- Reactive state -----------------------------------------------
    exp_reader = reactive.value(None)
    exp_names_updated = reactive.value(exp_names.copy())
    live_stream = reactive.value(None)
    live_status =reactive.value(None)

    # ---- Experiment selection / creation -------------------------------
    @reactive.calc
    def select_exp():
        return input.exp_name() not in (DEFAULT_MSG, NEW_EXP_MSG)

    @reactive.calc
    def new_exp():
        # input_submit_textarea only populates "new_exp_name" once the user
        # has explicitly submitted it, so reading it before that raises a
        # SilentException. Guard with the "in input" membership check.
        if "new_exp_name" not in input:
            return False
        return input.exp_name() == NEW_EXP_MSG and bool(input.new_exp_name())

    @reactive.calc
    def exp_name():
        if select_exp():
            return input.exp_name()
        if new_exp():
            return input.new_exp_name()
        return None

    @reactive.effect
    def initialize_experiment():
        name = exp_name()
        if name is None:
            # User backed out to the default/blank state: clear stale state
            # so the UI doesn't keep showing the previous experiment.
            exp_reader.set(None)
            return
        if new_exp():
            ExperimentTracker(name)  # creates the experiment on disk
        exp_reader.set(ExperimentReader(name))

    @reactive.effect
    def register_new_experiment():
        if new_exp():
            name = input.new_exp_name()
            choices = exp_names_updated.get().copy()
            if name not in choices:
                choices.insert(2, name)
                exp_names_updated.set(choices)
                ui.update_select("exp_name", choices=choices, selected=name)

    # ---- Derived experiment info ---------------------------------------
    @reactive.calc
    def reader_initialized():
        return exp_reader.get() is not None

    @reactive.calc
    def list_runs():
        return exp_reader.get().list_runs() if reader_initialized() else []

    @reactive.calc
    def empty_experiment():
        return len(list_runs()) == 0

    @reactive.calc
    def runs_summary_table():
        if not empty_experiment():
            return exp_reader.get().summarize_runs()
        return None

    @reactive.calc
    def launch_mode():
        return input.launch_or_inspect() if "launch_or_inspect" in input else None

    @reactive.calc
    def show_launcher():
        return new_exp() or (select_exp() and empty_experiment()) or launch_mode() == "Launch new Run"

    @reactive.calc
    def is_done():
        stream = live_stream.get()
        if stream is None:
            return False
        else:
            return stream.done()

    @reactive.calc
    def live_metrics():
        stream = live_stream.get()

        if stream is None:
            return pd.DataFrame()
        live_status.set(stream.get_status())

        if not is_done():
            reactive.invalidate_later(1)
        return stream.poll()


    # ---- Renders ---------------------------------------------------------
    @render.ui
    def launch_or_inspect_checkbox():
        if not empty_experiment():
            return ui.input_radio_buttons(
                "launch_or_inspect",
                None,
                ["Inspect Run", "Launch new Run"],
            )

    @render.ui
    def runs_inspector():
        if empty_experiment() or launch_mode() == "Launch new Run":
            return None
        return ui.TagList(
            ui.h5("Select a run:"),
            ui.output_data_frame("summary_runs"),
        )

    @render.data_frame
    def summary_runs():
        if not empty_experiment() and launch_mode() == "Inspect Run":
            return render.DataGrid(runs_summary_table(), selection_mode="row")
        return None  # explicit: nothing to show outside "Inspect Run" mode

    @render.text
    def d_model_value():
        V = input.vocab_size()
        a_d = input.alpha_dim()
        if V is not None and a_d is not None:
            V = int(V)
            return f"d_model = {int(V*a_d)}"
        else:
            return None


    @render.text
    def seq_len_value():
        V = input.vocab_size()
        a_L = input.alpha_length()
        if V is not None and a_L is not None:
            V = int(V)
            return f"L = {int(V*a_L)}"
        else:
            return None

    @render.text
    def K_value():
        V = input.vocab_size()
        rho = input.rho()
        if V is not None and rho is not None:
            V = int(V)
            return f"K = {int(V*rho) }"
        else:
            return None

    @render.text
    def lr_value():
        a_lr = input.alpha_lr()
        V = input.vocab_size()
        V_base = 128
        if V is not None and a_lr is not None:
            V = int(V)
            return f"lr = {a_lr*V_base/V :.5f}"

    @render.text
    def total_steps_value():
        a_steps = input.alpha_steps()
        V = input.vocab_size()
        if V is not None and a_steps is not None:
            V = int(V)
            return f"n_steps = {int(V)*a_steps}"

    @render.ui
    def runs_launcher():
        if show_launcher():
            trainer_args = TrainerArgs()
            model_params = ui.TagList(
                ui.h4("Model"),
                ui.input_numeric("vocab_size","V: Vocabulary Size",value = trainer_args.model_args.vocab_size),
                ui.input_numeric("alpha_dim", "alpha_d = d_model / V", value = trainer_args.model_args.alpha_dim),
                ui.input_numeric("alpha_length", "alpha_L = seq_len / V", value = trainer_args.model_args.alpha_length),
                ui.output_text("d_model_value"),
                ui.output_text("seq_len_value")                                
            )

            data_params = ui.TagList(
                ui.h4("Data"),
                ui.input_numeric("rho", "rho = K/V", value=trainer_args.data_args.rho),
                ui.input_numeric("batch_size","Batch Size", value=trainer_args.data_args.batch_size),
                ui.input_numeric("test_size","Test Size", value = trainer_args.data_args.test_size),
                ui.output_text("K_value")
            )

            optim_params = ui.TagList(
                ui.h4("Optimizer"),
                ui.input_numeric("alpha_lr","alpha_lr = lr * (V / 128)",value=trainer_args.optim_args.alpha_lr),
                ui.output_text("lr_value")
            )

            extra_params = ui.TagList(
                ui.h4("Run"),
                ui.input_numeric("alpha_steps","alpha_steps = total_steps/V",value=trainer_args.extra_args.alpha_steps),
                ui.input_numeric("n_prints","n_prints_scalars",value=trainer_args.extra_args.n_prints),
                ui.input_numeric("n_prints_model","n_prints_artifacts",value=trainer_args.extra_args.n_prints_model),
                ui.input_switch("fix_seed","Fix Seed?",False),
                ui.panel_conditional(
                    f"input.fix_seed",
                    ui.input_numeric("seed","seed",value=None)
                    ),
                ui.input_switch("enable_control","Enable Control?",False),
                ui.input_switch("enable_rewind","Enable Rewind",False),
                ui.output_text("total_steps_value")
            )

            launcher = ui.layout_columns(
                    model_params,
                    ui.TagList(data_params, optim_params),
                    extra_params,
                    ui.input_action_button("launch_btn", "Launch run", class_="btn-primary"),
                    col_widths={'sm':(3,3,3,3)}
                ) 
            return ui.accordion(ui.accordion_panel(ui.h4("Launcher Interface - Select the parameters for the Run"), launcher, value='launch_accordion'))

    @render.ui
    def live_dash():
        if input.launch_btn():
            return ui.TagList(
                ui.output_text("status_text"),
                ui.output_data_frame('live_metrics_stream')
            )

    @render.data_frame
    def live_metrics_stream():
        if input.launch_btn():
            return render.DataGrid(live_metrics())
        return None  # explicit: nothing to show outside "Inspect Run" mode

    @render.text
    def tmp():
        msg = f"{select_exp() = }, {new_exp() = }, {reader_initialized() = }, {empty_experiment() = } , {input.launch_btn() = }"
        if reader_initialized():
            msg += f", {exp_reader.get().experiment_name = }, {list_runs() = }"
        return msg

    @render.text
    def status_text():
        return str(live_status.get())

    # ---- launch ----
    @reactive.effect
    @reactive.event(input.launch_btn)
    def _launch():
        # Predict the run that ExperimentTracker will create and initialize MetricsStream
        reader = exp_reader.get()
        new_run_id = reader.next_run_id()
        live_stream.set(reader.get_metrics_stream(new_run_id))

        # Launch subprocess
        launcher_path = Path(__file__).parent / "launcher.py"

        names = {
            'model_args' : ['vocab_size','alpha_dim','alpha_length'],
            'data_args' : ['rho','batch_size','test_size'],
            'optim_args': ['alpha_lr'],
            'extra_args': ['alpha_steps','n_prints','n_prints_model','seed','enable_control','enable_rewind']
        }

        overrides = [
        f"{arg_type}.{name}={getattr(input, name)()}"
        for arg_type, names_list in names.items()
        for name in names_list
        ]

        cmd = [sys.executable, str(launcher_path),
                f"extra_args.experiment_name={exp_name()}", *overrides]
        subprocess.Popen(cmd)
        # block live_metrics() to poll data from the run previous to the 





app = App(app_ui, server)