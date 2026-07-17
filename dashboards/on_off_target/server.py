import numpy as np
# import torch
import matplotlib.pyplot as plt
from shiny import reactive, ui, render#, output
from tracklab import ExperimentReader
from configurations import apply_general_styles, set_font_sizes, create_fig

from utils import make_line_plot

apply_general_styles()
set_font_sizes(conf='tight')

def gauss_pdf(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)


experiment_name = 'logits'
reader = ExperimentReader(experiment_name,base_dir='../../data')
list_runs = reader.list_runs()


# -----------------------------
# Server
# -----------------------------
def server(input, output, session):
    # --- reactive derived state (clean extension point) ---
    @reactive.calc
    def available_runs():
        return reader.list_runs()

    # --- update dependent dropdown ---
    @reactive.effect
    def _update_runs():
        runs = available_runs()

        ui.update_select( 
            "run_id",
            choices=runs,
            selected=runs[-1] if runs else None,
        )

    # --- reactive derive number of steps ---
    @reactive.calc
    def num_matrix_steps():
        run_id = input.run_id()
        cfg = reader.load_config(run_id)
        return cfg['extra_args']['n_prints_model']


    # --- update dependent slider ---
    @reactive.effect
    def _update_matrix_steps():
        n_steps = num_matrix_steps()
        if n_steps is None:
            return
        ui.update_slider(
            "matrix_step",
            min=0,
            max=n_steps-1,
            value=0,
            step=1,
        )
    
    @reactive.effect
    def _update_comment_config_file():
        run_id = input.run_id()
        cfg = reader.load_config(run_id)['extra_args']
        
        comment = input.comment()
        if comment:
            cfg['comments'] = input.comment()
            reader.update_config(run_id, {'extra_args': cfg})
    
    @reactive.calc
    def _get_matrix_step_frac():
        num_matrix_stps = num_matrix_steps()
        matrix_step_idx = input.matrix_step()
        return matrix_step_idx / (num_matrix_stps-1) if num_matrix_stps > 0 else 0
    
    # --- extract data for plotting from experiment and run selection ---
    @reactive.calc
    def _get_plot_data():
        run_id = input.run_id()
        df = reader.load_metrics(run_id)
        table = df.pivot(index='step', columns='metric', values='value').reset_index()
        time = table['step']
        return time, table
    
    # --- extract summary table of parameters for experiment selection ---
    @reactive.calc
    def _get_summary_table():
        return reader.sumarize_runs()
    

    # --- extract list of artifacts ---
    @reactive.calc
    def _get_artifacts():
        run_id = input.run_id()
        return reader.list_artifacts(run_id)


    @reactive.calc
    def _get_attn1_data():
        run_id = input.run_id()
        list_artifacts = _get_artifacts()
        list_variable = list_artifacts[list_artifacts['file'].str.contains('attn1_histograms')]
        hists = [reader.load_artifact(run_id, file) for file in list_variable['file']]

        list_variable = list_artifacts[list_artifacts['file'].str.contains('attn1_matrix')]
        matrices = [reader.load_artifact(run_id, file) for file in list_variable['file']]
        return hists, matrices

    @reactive.calc
    def _get_attn2_data():
        run_id = input.run_id()
        list_artifacts = _get_artifacts()
        list_variable = list_artifacts[list_artifacts['file'].str.contains('attn2_histograms')]
        hists = [reader.load_artifact(run_id, file) for file in list_variable['file']]

        list_variable = list_artifacts[list_artifacts['file'].str.contains('attn2_matrix')]
        matrices = [reader.load_artifact(run_id, file) for file in list_variable['file']]
        return hists, matrices
    
    @reactive.calc
    def _get_logit_data():
        run_id = input.run_id()
        list_artifacts = _get_artifacts()
        list_variable = list_artifacts[list_artifacts['file'].str.contains('logits_histograms')]
        hists = [reader.load_artifact(run_id, file) for file in list_variable['file']]

        list_variable = list_artifacts[list_artifacts['file'].str.contains('logits_matrix')]
        matrices = [reader.load_artifact(run_id, file) for file in list_variable['file']]

        list_variable = list_artifacts[list_artifacts['file'].str.contains('logits_ind')]
        log_ind = [reader.load_artifact(run_id, file) for file in list_variable['file']]
        return hists, matrices, log_ind 
    
      
    # --- example output (start extending here) ---
    @output
    @render.text
    def selection():
        return f"Run: {input.run_id()}"
    

    @output
    @render.data_frame
    def summary_table():
        summary = _get_summary_table()
        return summary  # your pandas DataFrame
    
    @output
    @render.plot
    def plot_performance():
        time, data = _get_plot_data()
        matrix_step = _get_matrix_step_frac()*data['step'].max()
        fig , axes = create_fig(ncols=1,nrows=2,layout='tight')
        
        ax = axes[0]
        make_line_plot(ax, time, data['loss'], label=r'$L$')
        make_line_plot(ax, time, data['L_eff'], label=r"$L_eff$",color='green',linestyle='--')
        ax.legend(frameon=False,fontsize=5)

        ax = axes[1]
        make_line_plot(ax, time, data['top1_accuracy'], label=r'Top 1 Acc.')
        ax.legend(frameon=False,fontsize=5)
        ax.set_xlabel('Step',fontsize=5)

        for ax in axes:
            ax.axvline(matrix_step, color='peru', linestyle='--', lw=0.5,alpha=0.7)

        return fig
    

    @output
    @render.plot
    def plot_logit_stats():
        time, data = _get_plot_data()
        matrix_step = _get_matrix_step_frac()*data['step'].max()
        fig , axes = create_fig(ncols=1,nrows=2,layout='tight')

        ax = axes[0]
        make_line_plot(ax, time, data['on_target_mean'],err=data['on_target_std'], fill=True)
        make_line_plot(ax, time, data['h_star'],color='green',linestyle='--')
        ax.text(0.05,0.9,'On Target',fontsize=5,transform=ax.transAxes)
        
        ax = axes[1]
        make_line_plot(ax, time, data['off_target_mean'],err=data['off_target_std'], fill=True)
        ax.text(0.05,0.9,'Off Target',fontsize=5,transform=ax.transAxes)
        ax.set_xlabel('Step',fontsize=5)

        for ax in axes:
            ax.axvline(matrix_step, color='peru', linestyle='--', lw=0.5,alpha=0.7)

        return fig
    
    @output
    @render.plot
    def plot_ord_params():
        time, data = _get_plot_data()
        matrix_step = _get_matrix_step_frac()*data['step'].max()
        fig , axes = create_fig(ncols=1,nrows=3,layout='tight')

        for i, param in enumerate(['m','Q','Gamma']):
            ax = axes[i]
            make_line_plot(ax, time, data[param],label=param)
            ax.legend(frameon=False,fontsize=5)
        ax.set_xlabel('Step',fontsize=5)

        for ax in axes:
            ax.axvline(matrix_step, color='peru', linestyle='--', lw=0.5,alpha=0.7)

        return fig
    

    @output
    @render.plot
    def plot_logits():

        # Get histograms and matrix data
        histograms , matrix, log_ind = _get_logit_data()
        matrix_step_idx = input.matrix_step()
        hist_data = histograms[matrix_step_idx]
        edges = hist_data['edges']
        dx = np.diff(edges)
        
        # Get logits stats data
        time, data = _get_plot_data()
        matrix_step = _get_matrix_step_frac()*data['step'].max()
        time_idx = int(np.argmin(np.abs(time - matrix_step)))
        # time_idx = max(time_idx-1, 0)  # Ensure time_idx is not negative
        m_star = data['on_target_mean'][time_idx]
        s_star = data['on_target_std'][time_idx]
        s = data['off_target_std'][time_idx]

        # Get logits_ind data
        logits_ind = log_ind[matrix_step_idx] # shape (num_indices, vocab_size)
        S = np.log(np.sum(np.exp(logits_ind[:,:-1]),axis=-1)) # shape (num_indices,)
        h_star = logits_ind[:,-1]

        # Index of the batch to plot, keep the same index unless the randomize button is pressed
        if not hasattr(plot_logits, "i_batch"): plot_logits.i_batch = 0 
        if input.randomize(): plot_logits.i_batch = np.random.randint(0, 5)
        i_batch = plot_logits.i_batch

        # Get the matrix data for the current step and batch
        matrix_data = matrix[matrix_step_idx][i_batch] # (seq_len, vocab_size)
        vmin,vmax = matrix[-1].min(), matrix[-1].max()
        x_vals = np.linspace(edges.max(),edges.min(),200)

        # Create the figure and axes for the plots
        fig , axes = create_fig(ncols=3,nrows=1,layout='tight',sharex=False)

        ax = axes[0]
        im = ax.imshow(matrix_data, aspect='auto', cmap='viridis',vmin=vmin,vmax=vmax)
        plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        ax.set_xlabel("V",fontsize=5,labelpad=0,loc='right')
        ax.set_ylabel("L",fontsize=5,labelpad=0,loc='top')

        # ax = axes[1]
        # ax.bar(edges[:-1], hist_data['all'], width=dx, align='edge', color='gray', alpha=0.5, edgecolor='black',label='All')
        # ax.legend(frameon=False,fontsize=5)
        # #Eliminate the left axis
        # ax.spines['left'].set_visible(False)
        # ax.set_yticks([])
        # ax.set_xticks([])

        ax = axes[1]
        ax.bar(edges[:-1], hist_data['off'], width=dx, align='edge', color='red', alpha=0.5, edgecolor='black',label='Off')
        ax.plot(x_vals, gauss_pdf(x_vals, 0, s), color='red', linestyle='-')
        ax.bar(edges[:-1], hist_data['on'], width=dx, align='edge', color='blue', alpha=0.5, edgecolor='black',label='On')
        ax.plot(x_vals, gauss_pdf(x_vals, m_star, s_star), color='blue', linestyle='-')
        ax.legend(frameon=False,fontsize=5)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])

        ax = axes[2]
        ax.scatter(S,h_star,alpha=0.2,marker='.')
        ax.set_xlabel(r"$\log S = \log \sum_{i=1}^{V-1} e^{h_i}$",fontsize=8)
        ax.set_ylabel(r"$h^*$",fontsize=9)

        return fig
    
    @output
    @render.plot
    def plot_attn1():

        # Get histograms and matrix data
        histograms , matrix = _get_attn1_data()
        matrix_step_idx = input.matrix_step()
        hist_data = histograms[matrix_step_idx]
        edges = hist_data['edges']
        dx = np.diff(edges)

        
        # Index of the batch to plot, keep the same index unless the randomize button is pressed
        if not hasattr(plot_attn1, "i_batch"): plot_attn1.i_batch = 0 
        if input.randomize(): plot_attn1.i_batch = np.random.randint(0, 5)
        i_batch = plot_attn1.i_batch

        # Get the matrix data for the current step and batch
        matrix_data = matrix[matrix_step_idx][i_batch] # (seq_len, vocab_size)
        min,max = matrix[-1].min(), matrix[-1].max()

        # Create the figure and axes for the plots
        fig , axes = create_fig(ncols=1,nrows=3,layout='tight',sharex=False)

        ax = axes[0]
        im = ax.imshow(matrix_data, aspect='auto', cmap='viridis',vmin=min,vmax=max)
        plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        ax.set_xlabel("L",fontsize=5,labelpad=0,loc='right')
        ax.set_ylabel("L",fontsize=5,labelpad=0,loc='top')

        ax = axes[1]
        ax.bar(edges[:-1], hist_data['all'], width=dx, align='edge', color='gray', alpha=0.5, edgecolor='black',label='All')
        ax.legend(frameon=False,fontsize=5)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])

        ax = axes[2]
        ax.bar(edges[:-1], hist_data['off'], width=dx, align='edge', color='red', alpha=0.5, edgecolor='black',label='Off')
        ax.bar(edges[:-1], hist_data['on'], width=dx, align='edge', color='blue', alpha=0.5, edgecolor='black',label='On')
        ax.legend(frameon=False,fontsize=5)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
    
        return fig

    
    @output
    @render.plot
    def plot_attn2():

        # Get histograms and matrix data
        histograms , matrix = _get_attn2_data()
        matrix_step_idx = input.matrix_step()
        hist_data = histograms[matrix_step_idx]
        edges = hist_data['edges']
        dx = np.diff(edges)

        
        # Index of the batch to plot, keep the same index unless the randomize button is pressed
        if not hasattr(plot_attn2, "i_batch"): plot_attn2.i_batch = 0 
        if input.randomize(): plot_attn2.i_batch = np.random.randint(0, 5)
        i_batch = plot_attn2.i_batch

        # Get the matrix data for the current step and batch
        matrix_data = matrix[matrix_step_idx][i_batch] # (seq_len, vocab_size)
        min,max = matrix[-1].min(), matrix[-1].max()

        # Create the figure and axes for the plots
        fig , axes = create_fig(ncols=1,nrows=3,layout='tight',sharex=False)

        ax = axes[0]
        im = ax.imshow(matrix_data, aspect='auto', cmap='viridis',vmin=min,vmax=max)
        plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        ax.set_xlabel("L",fontsize=5,labelpad=0,loc='right')
        ax.set_ylabel("L",fontsize=5,labelpad=0,loc='top')

        ax = axes[1]
        ax.bar(edges[:-1], hist_data['all'], width=dx, align='edge', color='gray', alpha=0.5, edgecolor='black',label='All')
        ax.legend(frameon=False,fontsize=5)

        ax = axes[2]
        ax.bar(edges[:-1], hist_data['off'], width=dx, align='edge', color='red', alpha=0.5, edgecolor='black',label='Off')
        # ax.bar(edges[:-1], hist_data['on'], width=dx, align='edge', color='blue', alpha=0.5, edgecolor='black',label='On')
        ax.legend(frameon=False,fontsize=5)

        return fig