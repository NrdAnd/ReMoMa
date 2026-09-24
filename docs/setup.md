# Installation and environment

## Supported baseline

The reproducibility baseline uses Python 3.10/3.11, PyTorch 2.2.0, PyG 2.7.0, and NumPy 1.26.4. Direct runtime dependencies are pinned in `pyproject.toml`; transitive dependencies are not fully locked. The local validation environment is recorded in [validation](validation.md).

Both model families use PyTorch and PyG in the shared pipeline. CUDA is optional. The scripts select CUDA when PyTorch reports it available and otherwise use CPU. Apple MPS is not selected by these scripts.

## Virtual environment

From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

On Windows, activate with `.venv\Scripts\activate`. Alternatively, create a Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate remoma
python -m pip install -e '.[dev]'
```

`scripts/setup.sh` installs into the currently selected Python environment. It does not create or activate an environment.

## CPU and NVIDIA CUDA wheels

For Linux CPU execution, install the PyTorch CPU wheel before installing the project:

```bash
python -m pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

For a compatible NVIDIA environment using the historical CUDA 12.1 wheel:

```bash
python -m pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[dev]'
```

Select wheels according to the host platform and NVIDIA driver. These commands follow the [official previous-version installation instructions](https://pytorch.org/get-started/previous-versions/).

The tested operators work without `torch-scatter` or `torch-sparse`. Optional compiled extensions must match the exact PyTorch/CUDA combination; see [PyG 2.7 installation](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html). Do not install arbitrary extension wheels into an existing environment.

```bash
python -c 'import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__, torch.cuda.is_available())'
python -m unittest discover -s tests -v
python scripts/check_repository.py
```

## Data setup

Create `data/raw/` and place the original matching orderbook files there. Filenames must end in `_orderbook_10.csv`. Lower-level configurations without an explicit `data.raw_files` list use lexical filename order for zero-based file indices; their main examples expect at least five files: train `[0, 1, 2]`, validation `[3]`, test `[4]`. The complete pipeline instead selects explicit ISO dates and freezes the resolved file list in each run.

Use a private complete configuration under `configs/local/` to change paths, file indices, sampling, or hardware settings. All paths inside YAML configurations are relative to the repository root; absolute paths are supported. Editable installs locate that root automatically. For a wheel installation, set `REMOMA_ROOT` to the checkout path before importing the package, or run from the checkout root.

Order-flow features additionally require exact matching `_message_10.csv` filenames and equal row counts. The loader rejects mismatched pairs. Equal counts alone cannot establish alignment after files have been modified; use original paired exports.

## Research notebooks

```bash
python -m pip install -e '.[research]'
jupyter lab
```

Some exploratory notebooks import CuPy/cuDF and require a separate compatible RAPIDS environment. These GPU dependencies are not required for model training and are not installed by the research extra. Notebook parameter cells must be reviewed for input/output paths before execution. See [notebook guidance](../notebooks/README.md).
