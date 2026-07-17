from shiny import ui
from tracklab import ExperimentReader

# shiny run --reload --launch-browser scripts.py

experiment_name = 'logits'
reader = ExperimentReader(experiment_name,base_dir='../../data')
list_runs = reader.list_runs()

# -----------------------------
# UI
# -----------------------------
app_ui = ui.page_fluid(
    ui.tags.script(
        """
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }
        };
        """
    ),
    ui.tags.script(src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"),

    ui.h2("Experiment dashboard"),

    # SIDEBAR
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select(
                "run_id",
                "Select Run ID",
                choices=list_runs,
                selected=list_runs[-1] 
            ),
            ui.input_slider(
                "matrix_step",
                "Matrix Screenshots",
                min=0,
                max=1,
                value=0,
                step=1,

            ),
            # Button to get a random number each time is pressed
            ui.input_action_button("randomize", "Randomize"),

            # Text as input for a comment
            ui.input_text("comment", "Comment", value="")
        ),
    

    # MAIN PANEL
    ui.div(
        
        # row 1
        ui.row(
            ui.column(
                4,
                ui.card(
                    ui.card_header("Performance"),
                    ui.output_plot("plot_performance"),
                    style="height: 320px;"
                )
            ),
            ui.column(
                4,
                ui.card(
                    ui.card_header("Logit Stats"),
                    ui.output_plot("plot_logit_stats"),
                    style="height: 320px;"
                )
            ),
            ui.column(
                4,
                ui.card(
                    ui.card_header("Order Params."),
                    ui.output_plot("plot_ord_params"),
                    style="height: 320px;"
                )
            )
        ),
        # row 2
        ui.row(
            # ui.column(
            #     4,
            #     ui.card(
            #         ui.card_header("Attn 1"),
            #         ui.output_plot("plot_attn1"),
            #         style="height: 420px;",
            #         full_screen=True
            #     )
            # ),
            # ui.column(
            #     4,
            #     ui.card(
            #         ui.card_header("Attn 2"),
            #         ui.output_plot("plot_attn2"),
            #         style="height: 420px;",
            #         full_screen=True
            #     )
            # ),
            ui.column(
                12,
                ui.card(
                    ui.card_header("Logits"),
                    ui.output_plot("plot_logits"),
                    style="height: 420px;",
                    full_screen=True
                )
            )
        ),

        

        ui.row(
            ui.column(
                8,
                ui.card(
                    ui.card_header("Experiment Summary"),
                    ui.output_data_frame("summary_table"),
                    # style="height: 350px;"
                )
            ),
            ui.column(
                4,
                ui.card(
                    ui.card_header("Order Parameters"),
                    ui.div(
                        ui.markdown("""
                                $$M = \sum_{\mu=2}^L p_\mu^\\top W_{QK}^{(1)} p_{\mu-1}$$
                                $$Q = \dfrac{1}{K} \sum_{a \in \mathcal{T}} E(a)^\\top W_{QK}^{(2)}W_{OV}^{(1)} E(a) $$
                                $$\Gamma = \dfrac{1}{V-K} \sum_{a \in \overline{\mathcal{T}}} E(a)^\\top W_{OV}^{(2)} U(a)$$
                                $$h^* = \dfrac{\\beta}{r^2 V} M  Q  \Gamma $$
                                """),
                        style="font-size: 0.7rem;"
                    ),
                )
            )
        ),
    )
    )
)


