# Integration history and branch workflow

## Scope and recorded history

The current `main` branch contains the completed model-family integration. The temporary
`unified-model-pipeline` branch was used to prepare and validate that work; a separate remote
branch is no longer required. Removing the old branch pointers does not remove their commits,
because both source histories are reachable from `main` through the recorded merge commit.

The integration incorporated these source tips:

| Reference | Commit | Role |
| --- | --- | --- |
| Former GNN `main` | `10283d4` | Nine GNN architectures, result reports, initialization-seed support |
| Former `AdaptiveRecurrentSparseSTHNN` | `4ba8335` | Recurrent model, engineered features, threshold/ensemble/fold workflows |
| Shared ancestor | `5b038af` | Original common pipeline |
| Integration merge | `fc1a36a` | Two-parent merge containing both histories |

Before integration, the local `main` pointed to `5b038af`, nine commits behind the fetched GNN tip. Integration started from `10283d4`; neither source line was rewritten. The earlier `optimize-preprocessing` work was already part of the shared history. Subsequent organization, documentation, and end-to-end pipeline commits are descendants of `fc1a36a` on the current `main`.

The merge produced textual conflicts in `gnn/scripts/train.py` and `gnn/src/models/__init__.py`. They were resolved explicitly before structural migration. No blanket “ours” or “theirs” strategy was used.

## Resolution decisions

| Shared area | Integrated behavior |
| --- | --- |
| Model registry | Preserve all nine GNN types and `recurrent_sparse_sthnn` |
| Static batching | Preserve supported GCN/SAGE temporal variants; force tensor batching for recurrent; use PyG for GAT |
| CLI | Preserve model, hidden width, CNN width, learning rate, epochs, split, seed, and checkpoint-directory overrides |
| Evaluation | Preserve MCC, saved reports, threshold search, directional metrics, and threshold JSON |
| Sampling | Preserve a data-sampling seed independent of model initialization; include it in cache identity |
| Checkpoint directory | Explicit directory is exact; experiment runners pass it explicitly; ordinary runs get unique names |
| Engineered channels | Derive the input width consistently for both model families |
| Model-specific files | Separate family registries, implementations, and configurations |

A clean textual merge alone was insufficient: the old GNN checkpoint-directory suffix broke recurrent runners and accessed a missing `hidden_channels` key in recurrent configurations. The integrated code also addresses graph-cache identity, canonical node order, temporal-window overlap, training-only binner fitting, exact message-file pairing, and saved configuration provenance.

## Migration map

| Previous location | Current location |
| --- | --- |
| `gnn/src/` | `src/remoma/` |
| `gnn/src/models/{gcn,gat,sage,cgnn,stgcn,bin}.py` | `src/remoma/models/gnn/` |
| `gnn/src/models/recurrent_sparse_sthnn.py` | `src/remoma/models/recurrent/recurrent_sparse_sthnn.py` |
| `gnn/scripts/` | `scripts/`, with compatibility wrappers at the former paths |
| `gnn/config/default.yaml`, `test.yaml` | `configs/gnn/` |
| `gnn/config/recurrent*.yaml` | `configs/recurrent/` |
| `gnn/environment.yml`, `gnn/setup.sh` | `environment.yml`, `scripts/setup.sh` |
| `gnn/old_py/` | `archive/legacy/` |
| `LOB_Analysis/`, `glasso/`, TMFG notebooks | `notebooks/lob/`, `notebooks/glasso/`, `notebooks/tmfg/` |
| `tmfg/TMFG_core.py` | `src/remoma/graph/tmfg.py` |
| Versioned adjacency inputs | `data/graphs/` |
| GNN reports, slides, historical results | `docs/reports/` |
| Root heatmaps | One regenerated English figure in `docs/assets/`; originals retained in Git history |

Tracked bytecode, generated preprocessing/checkpoint fragments, temporary PDF renderings, and the generated similarity cache are excluded from the final source tree. Historical versions remain available through Git. Existing untracked local data and working files are not staged for publication.

Compatibility wrappers forward to the root implementation and translate old `config/<name>.yaml` arguments. New code must import `remoma`, not the former generic `src` namespace. Old external Python imports and arbitrary old relative data/output paths are not compatibility guarantees.

## Verify the preserved integration history

Start with a clean tracked working tree and `main` checked out:

```bash
git status --short
git switch main
git merge-base --is-ancestor 10283d4 HEAD
git merge-base --is-ancestor 4ba8335 HEAD
git show --no-patch --pretty=raw fc1a36a
git diff --check
python -m unittest discover -s tests -v
python scripts/check_repository.py
```

Each ancestry command must return exit status zero. `git show` must list `10283d4` and `4ba8335` as the two parents of `fc1a36a`. These checks use immutable commit identities and therefore remain valid even after the temporary remote branch names have been removed.

For the recorded source tips, the current `main` is already an integrated descendant. This statement applies to those exact commits and does not make claims about later work in another repository.

## Current branch workflow

New changes should start from the integrated `main` in the repository being edited:

```bash
git switch main
git pull --ff-only origin main
git switch -c descriptive-feature-name
```

Open future pull requests against `main` and run the documented checks before merging. The historical integration merge `fc1a36a` must remain reachable; ordinary merge-policy choices for later feature branches do not change its two parents.

## If an older development line receives new work

Bring the relevant commits into a fresh feature branch, then merge them into `main` while resolving semantic changes as well as text conflicts. Avoid copying a pre-migration directory tree over the current package.

Where paths moved, map the change to the corresponding new file. Git may detect renames, but large translations and reorganizations can require manual resolution. Re-run model, preprocessing, checkpoint, and CLI checks after conflict resolution. Never resolve a conflict by dropping the other family's registration or duplicating the common trainer.

Future conflicts cannot be ruled out. Separate family ownership, one shared contract, isolated runtime artifacts, small pull requests, and frequent synchronization reduce their frequency and make failures detectable.

## Recovery

Before a merge is committed, `git merge --abort` restores the pre-merge tracked state when Git can do so safely. Do not run it after making unrelated worktree changes without preserving those changes first. After publication, use a reviewed revert or follow-up fix; do not rewrite shared history. A revert of a merge has special ancestry consequences and must be planned with the repository owners.
