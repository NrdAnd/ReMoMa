# RecurrentSparseSTHNN Full Experiment Report

This document summarizes the RecurrentSparseSTHNN experiments run on Cisco LOB
data using the NMI-mean TMFG graph. It explains the model, the role of the
`flat` class, the threshold calibration procedure, and the results obtained for
each experimental setup.

## 1. Common Experimental Setup

All experiments use the same model family:

```text
RecurrentSparseSTHNN
```

The model was provided as a recurrent sparse spatio-temporal architecture and
integrated as an independent model type in the existing GNN pipeline.

The graph is built from the Cisco NMI-mean matrix:

```text
csv_NMI_matrix/lob_similarity_nmi_rellag_lag150_bins2000_mean.csv
```

The TMFG adjacency is then exported in the labeled format expected by the
RecurrentSparseSTHNN model.

For lag 100:

```text
data/adjacency/recurrent_sparse_tmfg_lag100_bins2000_from_nmi_mean.tsv
nodes = 2 sides * 10 levels * 101 lags = 2020
```

For lag 150:

```text
data/adjacency/recurrent_sparse_tmfg_lag150_bins2000_from_nmi_mean.tsv
nodes = 2 sides * 10 levels * 151 lags = 3020
```

The common model settings are:

```yaml
model:
  type: recurrent_sparse_sthnn
  hidden_dim: 5
  message_iterations: 1
  readout_mode: all_lags
  readout_dropout: 0.15
  out_dim: 3
```

The target settings are:

```yaml
price_type: mid
label_mode: abs
threshold: 100
prediction_horizon: 50
```

The three output classes are:

```text
0 = down
1 = flat
2 = up
```

## 2. Meaning of the Flat Class

The `flat` class is not a graph component. It is one of the three prediction
labels.

For each sample, the future mid-price movement is measured over
`prediction_horizon = 50`. With `label_mode = abs` and `threshold = 100`, the
label is assigned as:

```text
down: future movement <= -100
flat: -100 < future movement < 100
up:   future movement >= 100
```

In LOBSTER-style integer price units, a threshold of `100` corresponds to one
tick when prices are scaled by 10000.

Therefore, `flat` means that the future price movement is smaller than one tick
in absolute value over the chosen prediction horizon.

This class is very frequent in the test set. For this reason, it is important
to monitor not only accuracy, but also:

```text
signal precision
signal recall
flat_to_signal rate
```

Where:

```text
signal = prediction is down or up
flat_to_signal = true flat sample predicted as down or up
```

A high `flat_to_signal` rate means the model is producing too many false
directional signals.

## 3. Decision Rules and Threshold Calibration

The model outputs probabilities:

```text
P(down), P(flat), P(up)
```

The simplest decision rule is `argmax`:

```text
predict the class with the highest probability
```

However, because the data are highly imbalanced and `flat` is dominant, argmax
often predicts too many directional signals. Therefore, a post-training
threshold calibration step is used.

The calibrated decision rule is:

```text
if P(down) >= threshold_down, predict down
else if P(up) >= threshold_up, predict up
else predict flat
```

If both down and up thresholds are satisfied, the class with the larger
probability is selected.

Thresholds are tuned on the validation set and then applied unchanged to the
test set.

### Standard Threshold Tuning

The standard tuning objective is:

```text
score = macro_f1
```

This is the cleanest objective when we want to avoid adding an extra preference
against directional signals.

### Penalized Threshold Tuning

The penalized objective is:

```text
score = macro_f1 - 0.25 * flat_to_signal_rate
```

This does not modify the model weights and does not affect training. It only
changes how the final thresholds are chosen on validation.

The purpose is to prefer threshold pairs that reduce false directional signals
on true `flat` samples.

## 4. Split Strategies

Two split strategies were tested.

### by_lag

In `by_lag`, samples from all days can appear across train, validation, and
test. This split is less strict because train and test can contain samples from
the same trading days.

It is useful as an internal benchmark, but it is not the strongest test of
temporal generalization.

### by_file

In `by_file`, entire days are separated:

```text
train: files [0, 1, 2]
val:   file [3]
test:  file [4]
```

This is stricter because the test set is a full unseen trading day.

It is the preferred split when evaluating whether the model generalizes to new
Cisco days.

## 5. Results by Experimental Setup

### 5.1 Lag 100, by_lag, Single Model

Config:

```text
config/recurrent_sparse_sthnn.yaml
```

Checkpoint:

```text
checkpoints/recurrent_sparse_sthnn_nmi_mean/best.pt
```

This was the first strong baseline using the NMI-mean graph with 100 lags.

Standard threshold tuning:

```text
threshold_down = 0.78
threshold_up   = 0.79

Accuracy        = 0.8769
Macro F1        = 0.6172
Weighted F1     = 0.8822
Signal precision= 0.4190
Signal recall   = 0.5129
Signal F1       = 0.4612
Signal rate     = 0.1258
Flat_to_signal  = 0.0814
```

Penalized threshold tuning:

```text
threshold_down = 0.83
threshold_up   = 0.80

Accuracy        = 0.8845
Macro F1        = 0.6202
Weighted F1     = 0.8867
Signal precision= 0.4432
Signal recall   = 0.4841
Signal F1       = 0.4627
Signal rate     = 0.1122
Flat_to_signal  = 0.0696
```

Interpretation:

```text
The penalized threshold rule is better for this setup. It keeps macro F1 high,
raises accuracy, improves signal precision, and reduces false directional
signals on flat samples.
```

## 5.2 Lag 100, by_lag, Probability Ensemble

The ensemble averages predicted probabilities from multiple independently
trained checkpoints before applying threshold tuning.

### Ensemble: seeds 42 + 123 + 777

Checkpoints:

```text
checkpoints/recurrent_sparse_sthnn_nmi_mean/best.pt
checkpoints/recurrent_sparse_sthnn_nmi_mean_seed123/best.pt
checkpoints/recurrent_sparse_sthnn_nmi_mean_seed777/best.pt
```

Standard threshold tuning:

```text
Accuracy        = 0.8791
Macro F1        = 0.6163
Weighted F1     = 0.8833
Signal precision= 0.4246
Signal recall   = 0.4987
Flat_to_signal  = 0.0773
```

### Ensemble: seeds 42 + 123

Checkpoints:

```text
checkpoints/recurrent_sparse_sthnn_nmi_mean/best.pt
checkpoints/recurrent_sparse_sthnn_nmi_mean_seed123/best.pt
```

Standard threshold tuning:

```text
threshold_down = 0.76
threshold_up   = 0.79

Accuracy        = 0.8764
Macro F1        = 0.6202
Weighted F1     = 0.8824
Signal precision= 0.4185
Signal recall   = 0.5229
Signal rate     = 0.1284
Flat_to_signal  = 0.0831
```

Penalized threshold tuning:

```text
threshold_down = 0.76
threshold_up   = 0.82

Accuracy        = 0.8796
Macro F1        = 0.6180
Weighted F1     = 0.8838
Signal precision= 0.4267
Signal recall   = 0.5013
Signal rate     = 0.1207
Flat_to_signal  = 0.0770
```

Interpretation:

```text
The ensemble does not provide a clear improvement over the best single model.
The 42+123 ensemble matches the best macro F1, but it has lower signal
precision and higher flat_to_signal than the single seed-42 penalized model.
The 777 checkpoint was weaker and did not help the ensemble.
```

## 5.3 Lag 100, by_file, Single Model

Config:

```text
config/recurrent_sparse_sthnn_by_file.yaml
```

Checkpoint:

```text
checkpoints/recurrent_sparse_sthnn_nmi_mean_by_file/best.pt
```

Standard threshold tuning:

```text
threshold_down = 0.79
threshold_up   = 0.75

Accuracy        = 0.8459
Macro F1        = 0.6115
Weighted F1     = 0.8561
Signal precision= 0.3995
Signal recall   = 0.5415
Signal F1       = 0.4597
Signal rate     = 0.1651
Flat_to_signal  = 0.1118
```

Interpretation:

```text
Lag 100 with by_file is more aggressive. It obtains higher directional recall
and a strong macro F1, but it also produces more false directional signals on
true flat cases.
```

## 5.4 Lag 150, by_file, Single Model

Config:

```text
config/recurrent_sparse_sthnn_lag150_by_file.yaml
```

Checkpoint:

```text
checkpoints/recurrent_sparse_sthnn_lag150_nmi_mean_by_file/best.pt
```

This is the most directly comparable setup with the previous 150-lag model
while also using the stricter day-held-out split.

Standard threshold tuning:

```text
threshold_down = 0.77
threshold_up   = 0.79

Accuracy        = 0.8639
Macro F1        = 0.6009
Weighted F1     = 0.8636
Signal precision= 0.4421
Signal recall   = 0.4377
Signal F1       = 0.4399
Signal rate     = 0.1216
Flat_to_signal  = 0.0764
```

Penalized threshold tuning:

```text
threshold_down = 0.78
threshold_up   = 0.79

Accuracy        = 0.8642
Macro F1        = 0.5976
Weighted F1     = 0.8632
Signal precision= 0.4423
Signal recall   = 0.4284
Signal F1       = 0.4352
Signal rate     = 0.1190
Flat_to_signal  = 0.0747
```

Interpretation:

```text
For lag150 by_file, the standard threshold rule is preferable. The penalized
rule slightly reduces flat_to_signal but also reduces recall and macro F1.
This setup is conservative and clean, with fewer false signals than lag100
by_file.
```

## 5.5 Lag 150, by_lag, Single Model

Config:

```text
config/recurrent_sparse_sthnn_lag150.yaml
```

Checkpoint:

```text
checkpoints/recurrent_sparse_sthnn_lag150_nmi_mean/best.pt
```

Standard threshold tuning:

```text
threshold_down = 0.77
threshold_up   = 0.80

Accuracy        = 0.8789
Macro F1        = 0.6065
Weighted F1     = 0.8815
Signal precision= 0.4229
Signal recall   = 0.4686
Signal F1       = 0.4446
Signal rate     = 0.1146
Flat_to_signal  = 0.0737
```

Penalized threshold tuning:

```text
threshold_down = 0.79
threshold_up   = 0.80

Accuracy        = 0.8813
Macro F1        = 0.6066
Weighted F1     = 0.8828
Signal precision= 0.4304
Signal recall   = 0.4572
Signal F1       = 0.4434
Signal rate     = 0.1099
Flat_to_signal  = 0.0697
```

Interpretation:

```text
For lag150 by_lag, penalized threshold tuning is slightly preferable. It keeps
macro F1 essentially unchanged while increasing accuracy, increasing signal
precision, and reducing flat_to_signal.
```

## 6. Summary Table

```text
Setup                       Decision    Accuracy  MacroF1  WeightedF1  SigPrec  SigRec  Flat_to_signal
lag100 by_lag                standard    0.8769    0.6172   0.8822      0.4190   0.5129  0.0814
lag100 by_lag                penalized   0.8845    0.6202   0.8867      0.4432   0.4841  0.0696
ensemble 42+123 by_lag       standard    0.8764    0.6202   0.8824      0.4185   0.5229  0.0831
ensemble 42+123 by_lag       penalized   0.8796    0.6180   0.8838      0.4267   0.5013  0.0770
ensemble 42+123+777 by_lag   standard    0.8791    0.6163   0.8833      0.4246   0.4987  0.0773
lag100 by_file               standard    0.8459    0.6115   0.8561      0.3995   0.5415  0.1118
lag150 by_file               standard    0.8639    0.6009   0.8636      0.4421   0.4377  0.0764
lag150 by_file               penalized   0.8642    0.5976   0.8632      0.4423   0.4284  0.0747
lag150 by_lag                standard    0.8789    0.6065   0.8815      0.4229   0.4686  0.0737
lag150 by_lag                penalized   0.8813    0.6066   0.8828      0.4304   0.4572  0.0697
```

## 7. Final Interpretation

The strongest internal result is:

```text
lag100 by_lag with penalized thresholds
Macro F1        = 0.6202
Accuracy        = 0.8845
Signal precision= 0.4432
Flat_to_signal  = 0.0696
```

However, this split is less strict because train, validation, and test contain
samples from the same trading days.

The most comparable result with the previous 150-lag setup is:

```text
lag150 by_file with standard thresholds
Macro F1        = 0.6009
Accuracy        = 0.8639
Signal precision= 0.4421
Flat_to_signal  = 0.0764
```

This setup uses 150 lags and a day-held-out test split, making it the cleanest
comparison with the previous model.

The main tradeoff observed is:

```text
lag100: more aggressive, higher recall, higher macro F1, more false signals
lag150: more conservative, cleaner signals, lower flat_to_signal
```

The ensemble experiments did not produce a clear improvement over the best
single model. Therefore, the single-model results remain the most interpretable
and defensible.

## 8. Recommended Results to Report

For best internal performance:

```text
lag100 by_lag, penalized thresholds
```

For the fairest comparison with the previous lag150 model:

```text
lag150 by_file, standard thresholds
```

For a lag150 result under the less strict by_lag split:

```text
lag150 by_lag, penalized thresholds
```

