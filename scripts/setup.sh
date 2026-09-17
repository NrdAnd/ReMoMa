#!/usr/bin/env bash
# Install the project into the Python environment selected by the caller.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m pip install -e '.[dev]'
python -c 'import torch, torch_geometric; print("PyTorch:", torch.__version__, "PyG:", torch_geometric.__version__, "CUDA:", torch.cuda.is_available())'
