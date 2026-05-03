import pandas as pd
import matplotlib.pyplot as plt

from tracklab import ExperimentReader


class ExperimentLoader:
    def __init__(self, experiment_name, base_dir="./data"):
        self.exp = ExperimentReader(experiment_name,base_dir)

    def list_runs(self):
        return self.exp.list_runs()

    def load_metrics(self, run_id):
        return self.exp.load_metrics(run_id)
    
    def load_config(self, run_id):
        return self.exp.load_config(run_id)
    
    def list_artifacts(self, run_id):
        return self.exp.list_artifacts(run_id)
        

def get_metric_all_runs(loader, metric_name):

    data = []

    for run_id in loader.list_runs():
        df = loader.load_metrics(run_id)
        df = df[df["metric"] == metric_name]
        df["run_id"] = run_id
        data.append(df)
    
    return pd.concat(data, ignore_index=True)


def plot_metric(df,ax):
    for run_id, group in df.groupby("run_id"):
        ax.plot(group["step"],group["value"],label=run_id)
    
    # ax.legend()
    ax.set_xlabel("Steps")
    ax.set_ylabel("Value")
    # ax.set_xscale("log")
    # return plt

if __name__ == "__main__":
    loader = ExperimentLoader("low_rank")
    metrics = ['loss_total','top1_accuracy','kl_b_total','kl_b_bigram']
    fig , axes = plt.subplots(ncols=len(metrics))
    print('init plot')
    for i, metric_name in enumerate(metrics):
        # metric_name = 'kl_b_bigram'
        ax = axes[i]
        ax.set_title(metric_name)
        df = get_metric_all_runs(loader,metric_name)
        plot_metric(df , ax)
    print('end plot')
    ax.legend()
    plt.show()