from tracklab import ExperimentReader


if __name__ == "__main__":
    print("Hello, TrackLab!")
    exp = ExperimentReader("tmp")
    runs_list = exp.list_runs()
    print(runs_list)
    run_id = runs_list[-1]
    list_artif = exp.list_artifacts(run_id)

    print(list_artif)

    cfg = exp.load_config(run_id)
    print(cfg)

    metrics = exp.load_metrics(run_id)
    print(metrics)