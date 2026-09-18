# Research status and next experiments

The integrated implementation supports both families, engineered node channels, validation-only threshold tuning, multiple seeds, and complete fixed or walk-forward execution with training-only NMI/TMFG construction. See the [GNN records](gnn_experiments.md) and [recurrent report](recurrent_experiments.html) for historical measurements.

The earlier working documents overstated comparisons with the external HLOB benchmark and interpreted small seed variability as statistical significance. The available records do not justify those claims: test periods and subsamples differ, most local results use one held-out day, and graph estimation dates are not fully recorded.

The next defensible experiments are:

1. Rebuild graphs from documented training-only data and use the corrected preprocessing schema.
2. Compare both families on identical files, labels, lag depth, features, graph provenance, and decision rules.
3. Evaluate more held-out trading days and report variability across days separately from initialization seeds.
4. Repeat the larger CGNN configurations across seeds without choosing models using the final test results.
5. Verify aligned original message files before evaluating incremental order-flow features.
6. Evaluate execution-aware outcomes separately if the research scope later includes trading decisions.

The existing records suggest useful temporal and capacity experiments, but they do not establish a universal best operator, a capacity ceiling, statistical significance across market regimes, or profitability.
