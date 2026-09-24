# Deployment and publication

## Supported execution model

Deployment here means running the Python research pipeline as a batch job on a workstation or compute host. There is no HTTP inference API, streaming market-data client, broker integration, online hidden-state service, or automated trading execution component.

Choose a compute host with sufficient memory and storage for the selected graph and dataset.

For a complete experiment from raw CSCO files, follow the [pipeline deployment commands](pipeline.md). That runner estimates NMI/TMFG from training dates, so no historical reference adjacency needs to be supplied. Install the optional CuPy backend only on a compatible CUDA host.

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

Copy the complete run directory and the corresponding processed cache, including the fitted binner and metadata. Indexed caches also reference shared raw binary and per-fold volume/flow arrays: retain these dependencies and update their paths in the cache metadata after relocation. Retain the graph snapshot unchanged. Update filesystem paths in `resolved_config.yaml`, including `data.raw_files`, to their destination locations. If raw CSV files are unavailable, set `data.allow_missing_raw: true` explicitly and retain the original raw manifests as provenance. Indexed raw binary arrays remain required in that mode.

```bash
python scripts/evaluate.py --checkpoint /absolute/run/best.pt \
  --config /absolute/run/resolved_config.yaml \
  --thresholds /absolute/run/thresholds.json
```

Checkpoints record architecture/feature/label settings and graph hash, not a complete software/container lock or optimizer state. A reproducibility bundle should also include the exact commit, dependency inventory, data hashes, and the declared evaluation protocol. Keep private paths and raw market data out of public release assets.

## Prepare a GitHub release

1. Verify the preserved integration ancestry and current `main` workflow in [merging](merging.md).
2. Run the checks in [development](development.md) and wait for CI on the actual PR head.
3. Confirm that only intended source, documentation, graph inputs, and curated reports are tracked.
4. Confirm both project authors agree to release their original work under [AGPL-3.0-only](../LICENSE). Review the [bundled third-party notices](../THIRD_PARTY_NOTICES.md), source data rights, and the redistributability of any release artifacts.
5. Review graph/data provenance and distinguish historical measurements from results reproduced on the release commit.
6. Merge the reviewed release changes into `main`, then tag that exact commit according to the owners' versioning policy.

Repository visibility, branch protection, tags, and releases are GitHub administration actions. The pipeline never changes them and never uploads private datasets. The project license does not grant rights to raw market data or supersede third-party asset licenses. AGPL-3.0-only permits commercial and private internal use; its source-sharing obligations apply under the conditions stated in the license, including distribution and remote network interaction with a modified version.
