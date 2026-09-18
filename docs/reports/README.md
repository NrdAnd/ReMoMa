# Historical experiment reports

These files preserve research records from before the unified pipeline. They are not fresh measurements of the integration commit.

| Record | Description |
| --- | --- |
| [GNN experiments](gnn_experiments.md) | English presentation of the tracked classification results |
| [Research history](research_history.md) | Development sequence and earlier observations |
| [Research status](research_status.md) | Open questions and methodological corrections |
| [Recurrent experiments](recurrent_experiments.html) | Existing extended recurrent report, retained in English |
| [Implementation slides](implemented_solution.pptx) | Historical English presentation |
| [Raw result summaries](results/summary.csv) | Original numerical GNN records |

The raw text/CSV result files are preserved without recomputing metrics. Slides and the recurrent HTML document reflect their original experiments and may contain earlier interpretations. Current scientific qualifications below take precedence over historical claims.

## Interpretation limits

- Most GNN comparisons use one held-out day from a five-day CSCO dataset.
- July and September runs used different 50,000-sample subsets of that day.
- Initialization-seed variability is not a confidence interval for variation across trading days.
- The supplied graph inputs do not carry sufficient source-date provenance to certify training-only estimation.
- Schema version 3 includes the corrected temporal purging and training-only binner ranges introduced in version 2, and adds indexed/materialized storage identities. Old results are not a reproduction of the current pipeline.
- Comparing values from different datasets, dates, label definitions, threshold policies, or parameter budgets does not establish state-of-the-art superiority.
- Directional classification metrics are not profit, execution quality, or a backtest with trading costs.

Recompute on a declared protocol before using these figures in a publication claim. Superseded Italian working notes remain accessible in Git history; the English documents consolidate them and correct unsupported inferences.
