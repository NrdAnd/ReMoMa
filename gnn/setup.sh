#!/bin/bash
# Setup script for LOB-GNN on Lightning AI (or any conda environment)
# Run once after cloning the repo:
#   bash setup.sh

set -e

echo "==> Creating conda environment 'lob-gnn'..."
conda env create -f environment.yml
conda activate lob-gnn

echo ""
echo "==> Detecting PyTorch and CUDA versions..."
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_RAW=$(python -c "import torch; v=torch.version.cuda; print(v if v else 'cpu')")

if [ "$CUDA_RAW" = "cpu" ]; then
    CUDA_TAG="cpu"
else
    CUDA_TAG="cu$(echo $CUDA_RAW | tr -d '.')"
fi

echo "   torch=${TORCH}  cuda=${CUDA_TAG}"

echo ""
echo "==> Installing torch-geometric..."
pip install torch-geometric

echo ""
echo "==> Installing PyG sparse/scatter extensions (optional but faster)..."
pip install torch-scatter torch-sparse \
    -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA_TAG}.html" || \
    echo "   (extensions skipped — torch-geometric still works without them)"

echo ""
echo "Done. Activate with:  conda activate lob-gnn"
