# Deployment and publication

## Supported execution model

Deployment here means running the Python research pipeline as a batch job on a workstation or compute host. There is no HTTP inference API, streaming market-data client, broker integration, online hidden-state service, or automated trading execution component.

Use a remote compute host for full-data training when the local workstation cannot support the graph and dataset size. No local full-data training is required to prepare or publish the repository.

Use the same environment and validated configuration as the experiment. Provision raw files or a verified processed cache, the correct graph, and a writable run directory. Install from the repository checkout because scripts and experiment configurations are repository assets; the Python wheel contains the reusable `remoma` package.

## CPU container

Build from the repository root:

```bash
docker build -f deploy/Dockerfile -t remoma:local .
docker run --rm remoma:local
```

The default command prints training help. Mount private data and output directories explicitly for a batch run:

```bash
mkdir -p data/processed runs
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/data/raw:/app/data/raw:ro" \
  -v "$PWD/data/processed:/app/data/processed" \
  -v "$PWD/runs:/app/runs" \
  remoma:local python scripts/train.py \
  --config configs/gnn/default.yaml --model cgnn \
  --checkpoint-dir /app/runs/cgnn_container
```

The Dockerfile installs a Linux CPU PyTorch wheel and runs as a non-root user by default. Host bind-mount permissions still apply. The recipe is not a GPU container. For CUDA, provision a compatible host environment using [setup](setup.md), and validate the selected GPU/runtime combination before full runs.

## Move a checkpoint to another machine

Copy the complete run directory and the corresponding processed cache, including the fitted binner and metadata. Retain the graph snapshot unchanged. Update only filesystem paths in `resolved_config.yaml` to their destination locations. If raw files are unavailable, set `data.allow_missing_raw: true` explicitly and retain the original raw manifests as provenance.

```bash
python scripts/evaluate.py --checkpoint /absolute/run/best.pt \
  --config /absolute/run/resolved_config.yaml \
  --thresholds /absolute/run/thresholds.json
```

Checkpoints record architecture/feature/label settings and graph hash, not a complete software/container lock or optimizer state. A reproducibility bundle should also include the exact commit, dependency inventory, data hashes, and the declared evaluation protocol. Keep private paths and raw market data out of public release assets.

## Prepare a GitHub release

1. Follow the ancestry checks and review flow in [merging](merging.md).
2. Run the checks in [development](development.md) and wait for CI on the actual PR head.
3. Confirm that only intended source, documentation, graph inputs, and curated reports are tracked.
4. Select a repository license with the owners; verify rights for third-party code and redistributable artifacts. This change does not invent a license or alter upstream notices.
5. Review graph/data provenance and distinguish historical measurements from results reproduced on the release commit.
6. Merge while preserving history as described in the integration procedure, then tag the reviewed release according to the owners' versioning policy.

The preparation task does not itself push branches, merge a GitHub pull request, create a release, or expose private datasets.
