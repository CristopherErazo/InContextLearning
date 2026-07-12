from shiny import ui

# shiny run --reload --launch-browser scripts.py
# experiments = ['full_rank', 'low_rank','new_test_low_rank']
experiments = ['minimal_model']


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
                "experiment",
                "Select Experiment",
                choices=list(experiments),
            ),
            ui.input_select(
                "run_id",
                "Select Run ID",
                choices=[],
            ),
            ui.input_slider(
                "matrix_step",
                "Matrix Screenshots",
                min=0,
                max=1,
                value=0,
                step=1,

            ),
            # Button to change or not scale to log
            ui.input_checkbox(
                "log_scale",
                "Log Scale for Hist",
                value=False
            ),
            # Button to get a random number each time is pressed
            ui.input_action_button("randomize", "Randomize")
        ),
    

    # MAIN PANEL
    ui.div(
        # row 1
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header("Plot 1"),
                    ui.output_plot("plot1"),
                    style="height: 320px;"
                )
            )
        ),

        # row 2
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header("Matrix / Large plot"),
                    ui.output_plot("matrix_plot")
                )
            )
        ),

        ui.row(
            ui.column(
                4,
                ui.card(
                    ui.card_header("Experiment Summary"),
                    ui.output_data_frame("summary_table"),
                    # style="height: 350px;"
                )
            ),
            ui.column(
                8,
                ui.card(
                    ui.card_header("Order Parameters"),
                    ui.output_text("selection"),
                    ui.markdown("""
                                
                                $$ m_1 = \dfrac{1}{L-1} \sum_{\mu=2}^L p_{\mu}^T \cdot W_{QK}^{(1)} \cdot p_{\mu-1} $$
                                $$ m_2 = \dfrac{1}{K} \dfrac{1}{\|W_{OV}^{(1)}\|} \sum_{t \in T} E(t)^T \cdot W_{QK}^{(2)} W_{OV}^{(1)} \cdot E(t)$$
                                $$ \gamma = \dfrac{1}{V} \sum_{t=1}^V U(t)^T \cdot W_{OV}^{(2)} \cdot E(t) $$
                                $$ \eta_1 = \|W_{QK}^{(1)}\|$$
                                $$ \eta_2 = \dfrac{1}{\|W_{OV}^{(1)}\|} \|W_{QK}^{(2)} W_{OV}^{(1)}\|$$
                                """),
                    #         This experiment uses the update rule:

                    #         $$x_{t+1} = Ax_t + Bu_1$$

                    #         where:

                    #         - $x_t$ is the state vector
                    #         - $A \\in \\mathbb{R}^{n \\times n}$
                    #         - $u_t$ is the control input
                    #         """),
                    # style="height: 350px;"
                )
            )
        ),
    )
    )
)


