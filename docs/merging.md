# Branch integration and migration

## Scope and recorded history

The integration branch is `unified-model-pipeline`. It incorporates these fetched source tips:

| Reference | Commit | Role |
| --- | --- | --- |
| `origin/main` | `10283d4` | Nine GNN architectures, result reports, initialization-seed support |
| `origin/AdaptiveRecurrentSparseSTHNN` | `4ba8335` | Recurrent model, engineered features, threshold/ensemble/fold workflows |
| Shared ancestor | `5b038af` | Original common pipeline |
| Integration merge | `fc1a36a` | Two-parent merge containing both histories |

The local `main` originally pointed to `5b038af`, nine commits behind `origin/main`. Integration started from `origin/main`; neither original branch was rewritten. The earlier `optimize-preprocessing` branch was already part of the shared history.

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

## Verify the current integration

Start with a clean tracked working tree and the integration branch checked out:

```bash
git status --short
git fetch --prune origin
git merge-base --is-ancestor origin/main HEAD
git merge-base --is-ancestor origin/AdaptiveRecurrentSparseSTHNN HEAD
git diff --check
python -m unittest discover -s tests -v
python scripts/check_repository.py
```

Each ancestry command must return exit status zero. This proves that the exact fetched branch tip is already contained in the integration branch. If a branch has advanced, the check deliberately fails: integrate and test the new commits before proceeding.

For the recorded source tips, a merge into the integrated descendant is already up to date. Merging the integration branch into either recorded source tip can fast-forward. This is a property of those commit histories, not a guarantee about future edits.

## Publish the integration for review

These commands are instructions for the repository owners; no push is required to prepare the local result.

```bash
git switch unified-model-pipeline
git fetch --prune origin
git merge-base --is-ancestor origin/main HEAD
git merge-base --is-ancestor origin/AdaptiveRecurrentSparseSTHNN HEAD
git push -u origin unified-model-pipeline
```

Open a pull request targeting `main`. To retain the ancestry property, use a **merge commit or fast-forward**, according to branch protection. A squash merge discards the integration commit's parent relationships and therefore loses the proof that both original branch tips are contained. Rebase-merging can likewise rewrite the recorded integration history.

After the integration is present on `origin/main`, an owner can synchronize the old recurrent branch if it has no independent newer commits:

```bash
git fetch origin
git switch AdaptiveRecurrentSparseSTHNN
git merge --ff-only origin/main
git push origin AdaptiveRecurrentSparseSTHNN
```

If `--ff-only` fails, stop that sequence and inspect the divergence. Do not force-push to manufacture a clean result.

## If the old branches receive new work

Keep an untouched source branch and merge its new commits into the integration branch, resolving semantic changes as well as text conflicts. For a long-lived feature branch, merge the integrated `main` into it before further development. Avoid copying its pre-migration directory tree over the new package.

Where paths moved, map the change to the corresponding new file. Git may detect renames, but large translations and reorganizations can require manual resolution. Re-run model, preprocessing, checkpoint, and CLI checks after conflict resolution. Never resolve a conflict by dropping the other family's registration or duplicating the common trainer.

Future conflicts cannot be ruled out. Separate family ownership, one shared contract, isolated runtime artifacts, small pull requests, and frequent synchronization reduce their frequency and make failures detectable.

## Recovery

Before a merge is committed, `git merge --abort` restores the pre-merge tracked state when Git can do so safely. Do not run it after making unrelated worktree changes without preserving those changes first. After publication, use a reviewed revert or follow-up fix; do not rewrite shared history. A revert of a merge has special ancestry consequences and must be planned with the repository owners.
