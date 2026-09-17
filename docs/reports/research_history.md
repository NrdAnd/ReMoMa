# Research history

This English record consolidates the original working report. Values below are historical observations, not measurements repeated during integration. Consult [interpretation limits](README.md) before comparisons.

## Initial representation and training

The original experiments used five CSCO LOBSTER orderbook days, forty columns per snapshot, and a TMFG with 3,020 nodes and 9,054 undirected edges. The implementation moved feature construction from the training loop to memory-mapped arrays. It explored graph convolutions, positional lag features, normalization, graph pooling, temporal convolutions, and decision thresholds.

A significant early defect produced a constant zero volume channel: non-finite values propagated into percentile bin edges. The binner now filters non-finite fit values, maps non-finite transformed volumes to the lowest bin, and rejects degenerate quantile edges.

## Earlier reported progression

| Experiment stage | Reported test macro F1 | Qualification |
| --- | --- | --- |
| Volume-binning repair | 0.534 | Earlier baseline |
| Lag feature, LayerNorm, learning-rate adjustment | 0.610 | Percentage-label experiment |
| Mean-plus-max graph pooling | 0.629 | Percentage-label experiment |
| Temporal CNN followed by graph layers | 0.651 | Percentage-label experiment |
| CGNN with decision-threshold tuning | Approximately 0.628 | Absolute tick-label experiment |

These rows are not a single controlled ablation: target definitions and experimental settings changed. The original notes associated BatchNorm with unstable validation and used LayerNorm to remove dependence on running batch statistics. Mean-plus-max pooling retained both distributed and extreme node signals.

## Temporal architectures and capacity

CGNN applies temporal convolution along each side/level lag sequence before graph processing. STGCN interleaves temporal and graph processing. Early notes interpreted similar scores across models as an information ceiling near 0.62–0.63. Later capacity experiments improved recorded results, so that conclusion was withdrawn. No information-theoretic ceiling has been established.

A historical BiN experiment was reported to reduce macro F1 from 0.628 to 0.569. This is a result for that configuration, not evidence that the normalization is universally unsuitable.

## Labels and decisions

The target changed from relative price thresholds to an absolute one-tick threshold. In raw LOBSTER units, USD 0.01 is 100 units; using 0.01 directly was an earlier units error. The resulting class distribution was strongly dominated by flat samples.

Decision-threshold tuning kept the trained weights fixed and optimized down/up probability thresholds on validation. One historical CGNN comparison reported macro F1 0.522 with argmax and 0.628 with tuned thresholds. Both rules must be reported because their performance can generalize differently.

## Message alignment observations

The original report recorded these row counts:

| Date | Message rows | Orderbook rows | Equal counts |
| --- | ---: | ---: | --- |
| 2019-01-22 | 961786 | 849735 | No |
| 2019-01-23 | 933023 | 858811 | No |
| 2019-01-24 | 828838 | 828838 | Yes |
| 2019-01-25 | 700260 | 678638 | No |
| 2019-01-28 | 828085 | 828085 | Yes |

Equal counts alone do not prove correct event alignment. The unified loader rejects mismatched pairs and no longer falls back to pairing unrelated filenames by count. Order-flow channels were subsequently added in the recurrent branch and are now available through the shared feature pipeline.

The original notes reported mean recent signed trade flow of -347, -74, and +104 for future down, flat, and up classes respectively. This observational association was not rerun during integration and does not establish causality or incremental out-of-sample value.
