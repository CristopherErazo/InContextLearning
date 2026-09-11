# icl

In-context learning (induction heads) in a minimal two-layer linear-attention
transformer trained on a synthetic trigger-retrieval task.

Part of the `~/research` layout:

```
~/research/
├── lib/
│   ├── TrackLab/          file-based experiment tracker (sibling repo)
│   └── Rewind/            controllable training loop + dashboard (sibling repo)
└── projects/
    └── in-context-learning/
        └── code/          this repo
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). The two sibling libraries are pinned
to GitHub in `pyproject.toml`, so a fresh clone needs nothing else:

```bash
git clone https://github.com/CristopherErazo/InContextLearning.git 
cd InContextLearning
uv sync                 # .venv with icl, tracklab, rewind and torch from PyPI
```

### Torch build

By default torch comes from PyPI, which already ships CUDA wheels on Linux (the
cluster gets GPU support with no flags) and CPU-only wheels on Windows / macOS.
To pick a build explicitly, use one of the mutually exclusive extras:

```bash
uv sync --extra cpu     # small CPU-only wheels (laptops, CI)
uv sync --extra cu126   # CUDA 12.6 wheels (Windows GPU box, or pin CUDA on Linux)
uv sync --extra cu130   # CUDA 13.0 wheels (needs a recent NVIDIA driver)
```

### Developing the sibling libraries locally

`uv sync` installs `tracklab` and `rewind` from the pinned GitHub commits. To
work against the local checkouts in `~/research/lib/` instead, layer editable
installs on top after every `uv sync`:

```bash
uv sync
uv pip install -e ../../../lib/TrackLab -e ../../../lib/Rewind
```

Then run things with the activated `.venv` (`python ...`, `shiny run ...`) or
with `uv run --no-sync ...`. A plain `uv run` re-syncs the environment and puts
the pinned GitHub versions back.

To move the pin to a newer TrackLab / Rewind release, edit the `tag` (or `rev`)
in `[tool.uv.sources]` of `pyproject.toml`, then:

```bash
uv lock --upgrade-package tracklab   # or rewind
uv sync
```

## Usage

```bash
# one training run; any config field is overridable with OmegaConf dotted syntax
uv run python -u scripts/launcher.py model_args.vocab_size=512 extra_args.experiment_name=my_exp

# Rewind dashboard: launch form + live plot + pause / resume / rewind
uv run shiny run --reload scripts/dash.py

# sweep on the cluster (edit the arrays at the top of the script)
nohup bash shell/submit.sh > submit.log &
```

Run outputs go to `data/<experiment_name>/run_NNN/` (gitignored). Notebooks in
`notebooks/` read them back with `tracklab.ExperimentReader(exp, base_dir='../data')`.
