# Research notebooks

- [LOB similarity analysis](lob/README.md)
- [GLASSO experiments](glasso/README.md)
- [TMFG graph construction](tmfg/README.md)

These exploratory notebooks are separate from the supported model pipeline. They retain research logic and require experiment-specific data/path settings. Some require CuPy/cuDF and an NVIDIA RAPIDS environment. The common bootstrap locates the repository root and makes `remoma` importable.

Review every parameter cell before execution. Outputs and execution counts are cleared to keep commits reviewable and avoid embedding local data or stale findings. Save new figures and HTML under `runs/` and promote only reviewed results into `docs/reports/`.

The integration validates notebook structure and source syntax where applicable; it does not rerun GPU analyses or certify historical conclusions.
