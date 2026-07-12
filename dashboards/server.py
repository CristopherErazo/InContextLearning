import numpy as np
import torch
from shiny import reactive, ui, render#, output
from tracklab import ExperimentReader
from configurations import apply_general_styles, set_font_sizes, create_fig

apply_general_styles()
set_font_sizes(conf='tight')




# -----------------------------
# Server
# -----------------------------
def server(input, output, session):

    # --- reactive derived state (clean extension point) ---
    @reactive.calc
    def available_runs():
        experiment_name = input.experiment()
        reader = ExperimentReader(experiment_name)
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
        experiment_name = input.experiment()
        if not input.run_id():
            return 
        run_id = input.run_id()
        reader = ExperimentReader(experiment_name)
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
    
    # --- extract data for plotting from experiment and run selection ---
    @reactive.calc
    def _get_plot_data():
        experiment_name = input.experiment()
        if not input.run_id():
            return
        run_id = input.run_id()
        reader = ExperimentReader(experiment_name)
        df = reader.load_metrics(run_id)
        table = df.pivot(index='step', columns='metric', values='value').reset_index()
        return table
    
    # --- extract summary table of parameters for experiment selection ---
    @reactive.calc
    def _get_summary_table():
        experiment_name = input.experiment()
        if not input.run_id():
            return
        run_id = input.run_id()
        reader = ExperimentReader(experiment_name)
        summary = reader.sumarize_runs()
        return summary
    
    # --- extract list of artifacts ---
    @reactive.calc
    def _get_artifacts():
        experiment_name = input.experiment()
        if not input.run_id():
            return
        run_id = input.run_id()
        reader = ExperimentReader(experiment_name)
        artifacts = reader.list_artifacts(run_id)
        return artifacts
    
    # --- extract batch from artifact ---
    @reactive.calc
    def _get_batch():
        experiment_name = input.experiment()
        if not input.run_id():
            return
        run_id = input.run_id()
        reader = ExperimentReader(experiment_name)
        list_artifacts = _get_artifacts()
        batch_file = list_artifacts['file'].iloc[-1]
        batch = reader.load_artifact(run_id, batch_file)
        return batch

    # --- get matrices for matrix plot ---
    @reactive.calc
    def _get_matrices():
        experiment_name = input.experiment()
        if not input.run_id():
            return
        run_id = input.run_id()
        reader = ExperimentReader(experiment_name)
        list_artifacts = _get_artifacts()
        keys = ['scores1', 'scores2', 'logits']
        matrices_data = {k : [] for k in keys}
        for key in ['scores1', 'scores2', 'logits']:
            matrix_files = list_artifacts[list_artifacts['file'].str.contains(key)]['file'].to_list()
            for matrix_file in matrix_files:
                M = reader.load_artifact(run_id, matrix_file)
                matrices_data[key].append(M)
        return matrices_data


    # --- example output (start extending here) ---
    @output
    @render.text
    def selection():
        return f"Experiment: {input.experiment()} | Run: {input.run_id()}"
    

    @output
    @render.data_frame
    def summary_table():
        summary = _get_summary_table()
        if summary is None:
            return None
        return summary  # your pandas DataFrame
    
    @output
    @render.plot
    def plot1():
        fig, axes = create_fig(ncols=5,size='double',h=0.45,layout='tight')
        table = _get_plot_data()
        if table is not None:
            matrix_step_idx = input.matrix_step()
            num_matrix_stps = num_matrix_steps()
            last_step = table['step'].max()
            matrix_step = last_step * matrix_step_idx / (num_matrix_stps-1) if num_matrix_stps > 0 else 0

            ax = axes[0]
            ax.plot(table['step'], table['loss'], label='L',color='darkblue')
            # ax.plot(table['step'], table['loss_eff'], label=r'$L_{eff}$',color='orange')
            ax_twin = ax.twinx()
            ax_twin.spines['right'].set_visible(True)
            ax_twin.plot(table['step'], table['top1_accuracy'], label='acc',color='olivedrab')
            ax.set_xlabel('Step',fontsize=8)
            ax.set_title('Performance',fontsize=8)
            # Joint legends to plot them together only on ax to avoid duplicates
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_twin.get_legend_handles_labels()
            ax.legend(h1+h2, l1+l2, loc='upper right',fontsize=6)
            # ax.set_ylabel('Loss',color='darkblue')
            # ax_twin.set_ylabel('Accuracy',color='olivedrap')
            ax.axvline(matrix_step, color='peru', linestyle='--', lw=0.5,alpha=0.7)

            ax = axes[1]
            ax.plot(table['step'], table['m1'], label=r'$m_1$',color='darkblue')
            # ax.plot(table['step'], table['center1'], label=r'$c_1$',color='darkred')
            ax_twin = ax.twinx()
            ax_twin.spines['right'].set_visible(True)
            ax_twin.plot(table['step'], table['eta1'], label=r'$\eta_1$',color='olivedrab')
            ax.set_xlabel('Step',fontsize=8)
            ax.set_title('Layer 1',fontsize=8)
            # Joint legends to plot them together only on ax to avoid duplicates
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_twin.get_legend_handles_labels()
            ax.legend(h1+h2, l1+l2, loc='lower right', fontsize=6)
            # ax.set_ylabel(r'$m_1$',color='darkblue')
            # ax_twin.set_ylabel(r'$\eta_1$',color='olivedrap')
            ax.axvline(matrix_step, color='peru', linestyle='--', lw=0.5,alpha=0.7)

            ax = axes[2]
            ax.plot(table['step'], table['m2'], label=r'$m_2$',color='darkblue')
            ax_twin = ax.twinx()
            ax_twin.spines['right'].set_visible(True)
            ax_twin.plot(table['step'], table['eta2'], label=r'$\eta_2$',color='olivedrab')
            ax.set_xlabel('Step',fontsize=8)
            ax.set_title('Layer 2',fontsize=8)
            # Joint legends to plot them together only on ax to avoid duplicates
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_twin.get_legend_handles_labels()
            ax.legend(h1+h2, l1+l2, loc='lower right', fontsize=6)
            # ax.set_ylabel(r'$m_2$',color='darkblue')
            # ax_twin.set_ylabel(r'$\eta_2$',color='olivedrap')
            ax.axvline(matrix_step, color='peru', linestyle='--', lw=0.5,alpha=0.7)

            ax = axes[3]
            ax.plot(table['step'], table['gamma'], label=r'$\gamma$',color='darkblue')
            ax_twin = ax.twinx()
            ax_twin.spines['right'].set_visible(True)
            ax_twin.plot(table['step'], table['eta_gamma'], label=r'$\eta_\gamma$',color='olivedrab')
            ax.set_xlabel('Step',fontsize=8)
            ax.set_title('Readout',fontsize=8)
            # Joint legends to plot them together only on ax to avoid duplicates
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_twin.get_legend_handles_labels()
            ax.legend(h1+h2, l1+l2, loc='lower right', fontsize=6)
            # ax.legend(loc='upper left')
            # ax.set_ylabel(r'$\gamma$')
            ax.axvline(matrix_step, color='peru', linestyle='--', lw=0.5,alpha=0.7)

            ax = axes[4]
            ax.plot(table['step'], table['P_L1'], label=r'$P_{L1}$',color='darkblue')
            ax.plot(table['step'], table['P_L2'], label=r'$P_{L2}$',color='darkred',ls='--')
            ax_twin = ax.twinx()
            ax_twin.spines['right'].set_visible(True)
            ax_twin.plot(table['step'], table['delta_L'], label=r'$\Delta L$',color='olivedrab')
            ax.set_xlabel('Step',fontsize=8)
            ax.set_title('Effective Values',fontsize=8)
            # Joint legends to plot them together only on ax to avoid duplicates
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_twin.get_legend_handles_labels()
            ax.legend(h1+h2, l1+l2, loc='lower right', fontsize=6)
            # ax.legend(loc='upper left')
            # ax.set_ylabel(r'$\gamma$')
            ax.axvline(matrix_step, color='peru', linestyle='--', lw=0.5,alpha=0.7)
        return fig
    #  "P_L1":             P_L1,
    #     "P_L2":             P_L2,
    #     "delta_L":
    @output
    @render.plot
    def matrix_plot():
        fig, axes = create_fig(ncols=3,nrows=2,size='double',h=0.3,layout='tight',sharex=False,sharey=False)
        matrices_data = _get_matrices()
        batch = _get_batch()
        matrix_step_idx = input.matrix_step()
        log_scale = input.log_scale()


        table = _get_plot_data()

        # Get i_batch as random integer between 0 and 10 each time the randomize button is pressed, but keep the same i_batch if the button is not pressed
        if not hasattr(matrix_plot, "i_batch"):
            matrix_plot.i_batch = 0
        if input.randomize():
            matrix_plot.i_batch = np.random.randint(0, 10)
        # i_batch = 0
        position = 10
        if matrices_data is not None and batch is not None and table is not None:
            num_matrix_stps = num_matrix_steps()
            last_step = table['step'].max()
            matrix_step = last_step * matrix_step_idx / (num_matrix_stps-1) if num_matrix_stps > 0 else 0
            # Get all columns of the table for the current matrix step (or the closest one if the exact step is not present)
            current_row = table.iloc[(table['step'] - matrix_step).abs().argsort()[:1]]
            
            i_batch = matrix_plot.i_batch
            ax = axes[0,0]
            M = matrices_data['scores1'][matrix_step_idx][i_batch]
            M = np.tril(M,k=-1)
            im = ax.imshow(M, aspect='auto', cmap='viridis')
            ax.set_title(r'$S^{(1)}_{\mu,\nu}$')
            fig.colorbar(im, ax=ax)

            ax = axes[1,0]
            mask = batch['mask'].cpu() # shape (batch_size, seq_len, seq_len)  
            print(mask[0].diagonal())
            # Mask also rows above position 5:
            mask = mask #& (torch.arange(mask.shape[1]) > position).unsqueeze(0).unsqueeze(0)
            data = matrices_data['scores1'][matrix_step_idx][mask].flatten()#[:,position,:position].flatten()
            im = ax.hist(data, bins=30, color='darkblue', alpha=0.7)
            # Get min and max of the values of the bars to set the limits of the y axis
            min_height, max_height = im[0].min(), im[0].max()
            
            ax.set_ylim(1, max_height)
            # Plot gaussian approximation of the histogram
            # Get range of x values for the histogram
            x_min, x_max = ax.get_xlim()
            x_values = np.linspace(x_min, x_max, 100)
            y_values = np.exp(-0.5*(x_values-current_row['center1'].values[0])**2 / current_row['eta1'].values[0]**2) * max_height  
            ax.plot(x_values, y_values, color='red', linestyle='-',lw=1) 
            ax.axvline(current_row['m1'].values[0], color='peru', linestyle='--', ymax=0.5)
            # Remove y axis and ticks
            # ax.yaxis.set_visible(False)
            if log_scale:
                ax.set_yscale('log')

            ax = axes[0,1]
            M = matrices_data['scores2'][matrix_step_idx][i_batch]
            M = np.tril(M)
            im = ax.imshow(M, aspect='auto', cmap='viridis')
            ax.set_title(r'$S^{(2)}_{\mu,\nu}$')
            fig.colorbar(im, ax=ax)

            ax = axes[1,1]
            
            data = matrices_data['scores2'][matrix_step_idx][mask].flatten()
            im = ax.hist(data, bins=30, color='darkblue', alpha=0.7)
             # Get min and max of the values of the bars to set the limits of the y axis
            min_height, max_height = im[0].min(), im[0].max()
            
            ax.set_ylim(1, max_height)
            # Plot gaussian approximation of the histogram
            # Get range of x values for the histogram
            x_min, x_max = ax.get_xlim()
            x_values = np.linspace(x_min, x_max, 100)
            y_values = np.exp(-0.5*x_values**2 / current_row['eta2'].values[0]**2) * max_height  
            ax.plot(x_values, y_values, color='red', linestyle='-',lw=1) 
            ax.axvline(current_row['m2'].values[0], color='peru', linestyle='--', ymax=0.5)
            # Remove y axis and ticks
            # ax.yaxis.set_visible(False)
            if log_scale:
                ax.set_yscale('log')

            ax = axes[0,2]
            M = matrices_data['logits'][matrix_step_idx][i_batch]
            im = ax.imshow(M, aspect='auto', cmap='viridis')
            ax.set_title(r'Logits $h_{\mu,\tau}$')
            fig.colorbar(im, ax=ax)

            ax = axes[1,2]
            mask = batch['mask'].cpu()
            data = matrices_data['logits'][matrix_step_idx].flatten()
            im = ax.hist(data, bins=30, color='darkblue', alpha=0.7)
             # Get min and max of the values of the bars to set the limits of the y axis
            min_height, max_height = im[0].min(), im[0].max()
            ax.set_ylim(1, max_height)
            
            # Plot gaussian approximation of the histogram
            # Get range of x values for the histogram
            x_min, x_max = ax.get_xlim()
            x_values = np.linspace(x_min, x_max, 100)
            y_values = np.exp(-0.5*x_values**2 / current_row['eta_gamma'].values[0]**2) * max_height  
            ax.plot(x_values, y_values, color='red', linestyle='-',lw=1) 
            ax.axvline(current_row['gamma'].values[0], color='peru', linestyle='--', ymax=0.5)
            # Remove y axis and ticks
            # ax.yaxis.set_visible(False)
            if log_scale:
                ax.set_yscale('log')
        return fig
            


            
# fig , axes = create_fig(ncols=ncols, size='double',h=0.2,w=1.5)

# for i , file in enumerate(attn1_files['file']):
#     attn = reader.load_artifact(run_id, file)[i_batch][:l,:l]
    
#     ax = axes[i]
#     # Mask the uper triangle of the score matrix to -inf for scores and to 0 for attn
#     # if not is_attn:
#         # attn = np.tril(attn)

#     im = ax.imshow(attn, aspect='auto', cmap='viridis')
#     if masked:
#         plot_mask_squares(ax, mask,lw=0.6,alpha=0.7,color='r')

    
#     ax.set_title(f'step={extract_number(file[-8:-4])}',fontsize=8)
#     fig.colorbar(im, ax=ax, fraction=0.03, pad=-0.18, shrink=0.5)


