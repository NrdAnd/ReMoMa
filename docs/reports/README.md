# Historical experiment reports

This archive contains the results, configurations, and analysis of experiments performed before the unified pipeline. The recorded metrics belong to those experiments; full-data runs with the current pipeline are still pending.

| Record | Description |
| --- | --- |
| [GNN experiments](gnn_experiments.md) | Recorded classification results |
| [Research history](research_history.md) | Development sequence and earlier observations |
| [Research status](research_status.md) | Open questions and planned comparisons |
| [Recurrent experiments](recurrent_experiments.html) | Extended recurrent experiment report |
| [Implementation slides](implemented_solution.pptx) | Presentation of the original experiments |
| [Raw result summaries](results/summary.csv) | Original numerical GNN records |

The original text/CSV metrics, slides, and recurrent HTML report are retained unchanged. Read their conclusions in the context of the protocols below.

## Evaluation protocols and scope

- Most GNN comparisons use one held-out day from a five-day CSCO dataset.
- July and September runs used different 50,000-sample subsets of that day.
- Seed variability describes repeated initialization on the same data, not variation across trading days.
- The source dates used to estimate the reference graphs are not fully recorded, so training-only estimation cannot be verified for those graphs. This does not establish that test data was used.
- Current preprocessing uses purged temporal splits, training-only volume bins, and indexed/materialized cache identities. These changes require new runs to measure current-pipeline performance.
- Model comparisons require matched data, dates, labels, decision rules, and parameter budgets. The reported classification metrics do not measure trading returns or costs.

Use a declared, matched protocol for new comparisons. Earlier working notes remain available in Git history.
