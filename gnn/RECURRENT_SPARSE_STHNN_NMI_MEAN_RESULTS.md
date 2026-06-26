# RecurrentSparseSTHNN on Cisco with NMI-mean Graph

This document summarizes the `RecurrentSparseSTHNN` run using the Cisco
NMI-mean matrix with 2000 bins and consecutive lags `0..100`.

## Model

The model is `RecurrentSparseSTHNN`, integrated as a separate model type in the
GNN pipeline. The network receives preprocessed LOB tensors with logical shape:

```text
[batch, nodes, features]
nodes = 2 sides * 10 levels * 101 lags = 2020
features = preprocessed price/volume features
```

The graph structure is obtained by rebuilding the TMFG directly from:

```text
csv_NMI_matrix/lob_similarity_nmi_rellag_lag150_bins2000_mean.csv
```

The final adjacency used by the model is:

```text
data/adjacency/recurrent_sparse_tmfg_lag100_bins2000_from_nmi_mean.tsv
```

Main config:

```text
config/recurrent_sparse_sthnn.yaml
```

Main model settings:

```yaml
model:
  type: recurrent_sparse_sthnn
  hidden_dim: 5
  message_iterations: 1
  readout_mode: all_lags
  readout_dropout: 0.15
  out_dim: 3
```

Model statistics:

```text
Trainable parameters: 34,528
Recurrent blocks: 160
Temporal edges: 11,101
Same-lag edges: 4,848
```

## Dataset and Target

Raw files used:

```text
CSCO_2019-01-22_34200000_57600000_orderbook_10.csv
CSCO_2019-01-23_34200000_57600000_orderbook_10.csv
CSCO_2019-01-24_34200000_57600000_orderbook_10.csv
CSCO_2019-01-25_34200000_57600000_orderbook_10.csv
CSCO_2019-01-28_34200000_57600000_orderbook_10.csv
```

Target settings:

```yaml
n_lags: 100
n_levels: 10
label_mode: abs
threshold: 100
prediction_horizon: 50
```

Preprocessed split:

```text
train raw: 2,975,867
val raw:     637,686
test raw:    637,689

train balanced: 150,000 samples, 50,000 per class
val:             25,000 samples
test:            50,000 samples
```

## Required Commands

Run from the `gnn` directory:

```bash
cd ~/ReMoMa/gnn
```

Preprocessing:

```bash
python scripts/preprocess_dataset.py --config config/recurrent_sparse_sthnn.yaml
```

Training:

```bash
python scripts/train.py --config config/recurrent_sparse_sthnn.yaml
```

Training is configured for up to 50 epochs. In this run, it stopped at epoch 20
because early stopping was triggered. The best checkpoint was saved at epoch
12.

Main training outputs:

```text
Best validation epoch: 12
Best validation loss: 0.5109
Best validation accuracy: 0.781
Best validation macro F1: 0.574
Early stopping: epoch 20
Checkpoint: checkpoints/recurrent_sparse_sthnn_nmi_mean/best.pt
Metrics: checkpoints/recurrent_sparse_sthnn_nmi_mean/metrics.csv
```

## Evaluation with Standard Thresholds

This command calibrates the decision thresholds on the validation split by
optimizing `macro_f1`, then evaluates the checkpoint on the test split:

```bash
python scripts/tune_threshold.py \
  --checkpoint checkpoints/recurrent_sparse_sthnn_nmi_mean/best.pt \
  --config config/recurrent_sparse_sthnn.yaml

python scripts/evaluate.py \
  --checkpoint checkpoints/recurrent_sparse_sthnn_nmi_mean/best.pt \
  --config config/recurrent_sparse_sthnn.yaml \
  --thresholds checkpoints/recurrent_sparse_sthnn_nmi_mean/thresholds.json
```

Selected thresholds:

```text
down >= 0.78
up   >= 0.79
```

Test results:

```text
Accuracy:    0.8769
F1 Macro:    0.6172
F1 Weighted: 0.8822

Signal precision: 0.4190
Signal recall:    0.5129
Signal F1:        0.4612
Signal rate:      0.1258
Flat->signal:     0.0814
```

Classification report:

```text
              precision    recall  f1-score   support

        down     0.4108    0.5656    0.4759      2509
        flat     0.9428    0.9186    0.9306     44863
          up     0.4289    0.4627    0.4452      2628

    accuracy                         0.8769     50000
   macro avg     0.5942    0.6490    0.6172     50000
weighted avg     0.8891    0.8769    0.8822     50000
```

Confusion matrix:

```text
[[ 1419  1090     0]
 [ 2032 41212  1619]
 [    3  1409  1216]]
```

## Recommended Conservative Evaluation

This is the recommended version for reporting, because it penalizes false
directional signals on true `flat` cases.

Command:

```bash
python scripts/tune_threshold.py \
  --checkpoint checkpoints/recurrent_sparse_sthnn_nmi_mean/best.pt \
  --config config/recurrent_sparse_sthnn.yaml \
  --objective macro_f1_penalized \
  --flat-fp-penalty 0.25 \
  --save-thresholds checkpoints/recurrent_sparse_sthnn_nmi_mean/thresholds_penalized.json

python scripts/evaluate.py \
  --checkpoint checkpoints/recurrent_sparse_sthnn_nmi_mean/best.pt \
  --config config/recurrent_sparse_sthnn.yaml \
  --thresholds checkpoints/recurrent_sparse_sthnn_nmi_mean/thresholds_penalized.json
```

Selected thresholds:

```text
down >= 0.83
up   >= 0.80
```

Test results:

```text
Accuracy:    0.8845
F1 Macro:    0.6202
F1 Weighted: 0.8867

Signal precision: 0.4432
Signal recall:    0.4841
Signal F1:        0.4627
Signal rate:      0.1122
Flat->signal:     0.0696
```

Classification report:

```text
              precision    recall  f1-score   support

        down     0.4514    0.5221    0.4842      2509
        flat     0.9403    0.9304    0.9353     44863
          up     0.4343    0.4479    0.4410      2628

    accuracy                         0.8845     50000
   macro avg     0.6087    0.6335    0.6202     50000
weighted avg     0.8892    0.8845    0.8867     50000
```

Confusion matrix:

```text
[[ 1310  1199     0]
 [ 1591 41739  1533]
 [    1  1450  1177]]
```

## Brief Interpretation

The conservative calibration improves the final result compared with both the
plain `argmax` decision rule and the standard threshold calibration:

```text
Argmax test macro F1:        0.5730
Tuned macro F1:              0.6172
Penalized tuned macro F1:    0.6202

Argmax signal precision:     0.2997
Tuned signal precision:      0.4190
Penalized signal precision:  0.4432

Argmax flat->signal:         0.1859
Tuned flat->signal:          0.0814
Penalized flat->signal:      0.0696
```

The model still produces some false positives on true `flat` cases, but the
penalized version substantially reduces this issue. Opposite-direction errors
are almost zero:

```text
true_down -> pred_up: 0
true_up   -> pred_down: 1
```

For this run, the best baseline to report is:

```text
checkpoint: checkpoints/recurrent_sparse_sthnn_nmi_mean/best.pt
thresholds: checkpoints/recurrent_sparse_sthnn_nmi_mean/thresholds_penalized.json
down >= 0.83
up   >= 0.80
```
