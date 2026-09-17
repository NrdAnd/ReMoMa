# GNN experiment records

The table below reproduces the numerical fields in the tracked result summary. It does not rerun experiments or average across potentially different test subsets. All listed records use the split stored in the source CSV. See [report qualifications](README.md) and the [original CSV](results/summary.csv).

| Timestamp | Model | Split | Rule | Macro F1 | MCC | Accuracy | Parameters |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 20260708_153555 | gcn | by_file | argmax | 0.4757 | 0.3107 | 0.6250 | 205651 |
| 20260708_153555 | gcn | by_file | tuned | 0.5715 | 0.3427 | 0.8360 | 205651 |
| 20260708_172107 | sage | by_file | argmax | 0.5445 | 0.3844 | 0.7214 | 181603 |
| 20260708_172107 | sage | by_file | tuned | 0.6086 | 0.3976 | 0.8629 | 181603 |
| 20260708_195533 | gat | by_file | argmax | 0.5874 | 0.3794 | 0.8282 | 188355 |
| 20260708_195533 | gat | by_file | tuned | 0.5841 | 0.3710 | 0.8246 | 188355 |
| 20260708_213729 | cgnn_sage | by_file | argmax | 0.5529 | 0.3803 | 0.7414 | 184707 |
| 20260708_213729 | cgnn_sage | by_file | tuned | 0.6008 | 0.3868 | 0.8484 | 184707 |
| 20260715_112522 | cgnn | by_file | argmax | 0.5379 | 0.3887 | 0.7075 | 181450 |
| 20260715_112522 | cgnn | by_file | tuned | 0.6246 | 0.4213 | 0.8699 | 181450 |
| 20260715_135605 | stgcn | by_file | argmax | 0.5800 | 0.4058 | 0.7684 | 183263 |
| 20260715_135605 | stgcn | by_file | tuned | 0.6195 | 0.4168 | 0.8702 | 183263 |
| 20260715_190724 | cgnn_sage | by_file | argmax | 0.5261 | 0.3787 | 0.6930 | 273325 |
| 20260715_190724 | cgnn_sage | by_file | tuned | 0.6153 | 0.4078 | 0.8690 | 273325 |
| 20260905_173716 | cgnn | by_file | argmax | 0.5414 | 0.3829 | 0.7118 | 181450 |
| 20260905_173716 | cgnn | by_file | tuned | 0.6237 | 0.4206 | 0.8564 | 181450 |
| 20260905_200905 | cgnn | by_file | argmax | 0.5442 | 0.3928 | 0.7154 | 181450 |
| 20260905_200905 | cgnn | by_file | tuned | 0.6295 | 0.4287 | 0.8694 | 181450 |
| 20260905_222628 | stgcn | by_file | argmax | 0.5632 | 0.4035 | 0.7411 | 183263 |
| 20260905_222628 | stgcn | by_file | tuned | 0.6315 | 0.4329 | 0.8793 | 183263 |
| 20260906_010756 | stgcn | by_file | argmax | 0.5652 | 0.4042 | 0.7411 | 183263 |
| 20260906_010756 | stgcn | by_file | tuned | 0.6337 | 0.4361 | 0.8784 | 183263 |
| 20260906_025031 | sage | by_file | argmax | 0.5319 | 0.3733 | 0.7020 | 271855 |
| 20260906_025031 | sage | by_file | tuned | 0.6253 | 0.4204 | 0.8633 | 271855 |
| 20260906_092737 | cgnn | by_file | argmax | 0.5209 | 0.3770 | 0.6762 | 720251 |
| 20260906_092737 | cgnn | by_file | tuned | 0.6428 | 0.4483 | 0.8752 | 720251 |
| 20260906_104356 | stgcn_sage | by_file | argmax | 0.5721 | 0.4096 | 0.7592 | 181899 |
| 20260906_104356 | stgcn_sage | by_file | tuned | 0.6324 | 0.4322 | 0.8716 | 181899 |
| 20260906_132212 | cgnn_gat | by_file | argmax | 0.5335 | 0.3440 | 0.7033 | 186387 |
| 20260906_132212 | cgnn_gat | by_file | tuned | 0.6057 | 0.3978 | 0.8387 | 186387 |
| 20260906_150243 | stgcn_gat | by_file | argmax | 0.5658 | 0.3951 | 0.7593 | 186371 |
| 20260906_150243 | stgcn_gat | by_file | tuned | 0.5975 | 0.3794 | 0.8505 | 186371 |

The largest recorded CGNN run reports tuned macro F1 0.6428 and MCC 0.4483. This is a historical single-run observation on the recorded test subset. Parameter budgets differ across rows, and July/September test subsamples were not identical. Raw per-run text reports are retained in the same results directory.

Earlier notes reported three-run means of 0.6259 ± 0.0031 for CGNN macro F1 and 0.6282 ± 0.0076 for STGCN. Those summaries combine historical runs and must not be interpreted as confidence intervals for future-day performance. Use a fresh matched protocol for formal comparisons.
