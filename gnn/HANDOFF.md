# HANDOFF — GNN su LOB CSCO vs benchmark HLOB

> Documento di passaggio (2026-07-08). Autosufficiente: una chat nuova può ripartire da qui senza il transcript.
> Progetto: predizione direzione mid-price (down/flat/up) da LOB CSCO (LOBSTER) con GNN su grafo TMFG.

---

## TL;DR

- Il nostro task ha **le stesse identiche label** del paper **HLOB** (Briola–Bartolucci–Aste): differenza mid-price punto-a-punto, soglia 1 tick, orizzonte 50. Quindi il confronto è **legittimo**.
- **HLOB su CSCO @ orizzonte 50: F1 macro ≈ 0.60, MCC ≈ 0.47** (il migliore di 10 modelli SOTA; l'intero campo si ferma lì).
- Sullo **split onesto (`by_file`) con tuning**, la nostra famiglia di modelli sta a **~0.61 F1**, cioè **alla pari sulla F1**, con ~30× meno dati. **Non siamo dietro sulla F1.**
- **MA** non abbiamo mai calcolato l'**MCC** (la metrica di punta di HLOB): è il buco che impedisce di dichiarare la parità. Ora è stato aggiunto al codice.
- Preliminarmente **nessuna architettura vince** (GCN ≈ SAGE ≈ GAT ≈ CGNN ≈ STGCN ≈ STHNN, tutte ~0.60): è un **soffitto informativo**, non un limite di architettura.

---

## 1. Contesto del progetto

- **Dati**: LOB CSCO (LOBSTER), 5 giorni. Prezzi in interi × 10000 (1 tick = $0.01 = 100 unità).
- **Grafo**: TMFG, **3020 nodi** (2 lati × 10 livelli × 151 lag), **18108 archi diretti**. Feature per nodo: `(p−mid)/mid`, volume quantile-binned, + feature lag ∈ [0,1] opzionale.
- **Target**: direzione mid-price a **k = 50** eventi, 3 classi (0=down, 1=flat, 2=up).
- **Label (config attuale, `config/default.yaml`)**:
  ```
  price_type: mid | label_mode: abs | threshold: 100 | prediction_horizon: 50
  Δm = mid[t+50] − mid[t]
  down se Δm ≤ −100 ; up se Δm ≥ +100 ; flat altrimenti  → ~90% flat
  ```
- **Training bilanciato** (`max_samples_per_class: 50000`); **val/test = distribuzione naturale** (subsample 25k/50k).
- **Split**: `by_lag` (default, ottimistico) vs `by_file` (onesto, giorni interi held-out: train [0,1,2] / val [3] / test [4]).
- **Modelli** (`src/models/`): `gcn`, `cgnn`, `stgcn` (tutti usano **GCNConv**; static path); `sage` (SAGEConv); `gat` (GATConv). Tutti con mean+max pool, LayerNorm, lag feature. ~180–205k parametri.
- **Vincoli fissi** (non negoziabili nel progetto): k=50 non si tocca; label mid/tick/threshold 100 restano; deve restare **GNN** (no sostituzione con LSTM puro).

### Ambiente
- **Training su Lightning AI (GPU L4)**, non in locale. In locale **non c'è torch** → non si può allenare né validare i .py importandoli (solo `py_compile` per la sintassi).
- Conseguenza operativa: **le modifiche al codice vanno sincronizzate su Lightning** (git push → pull) prima che un run le veda.

---

## 2. Il benchmark HLOB (il metro di paragone)

- **Paper**: A. Briola, S. Bartolucci, T. Aste — *"HLOB – Information Persistence and Structure in Limit Order Books"*, arXiv:2405.18938. Codice: **LOBFrame** (github.com/FinancialComputingUCL/LOBFrame).
- **HLOB** è un modello TMFG + Homological CNN + LSTM: stessa famiglia "TMFG-based" del nostro progetto.
- **Label identiche alle nostre** (Eq. 2 del paper): Δm punto-a-punto, θ = 1 tick, orizzonte ∈ {10,50,100}. Nella nota 7 **rifiutano esplicitamente lo smoothing FI-2010** → confronto valido sul task.
- **CSCO è un large-tick stock** (il gruppo dove questi modelli rendono meglio).
- **Protocollo**: training bilanciato (5k/classe/giorno); **test = distribuzione naturale su 10 giorni held-out** (= nostro `by_file`); metriche **mediate su 3 anni** (2017–2019, ~40 train + 5 val + 10 test giorni/anno). **F1 = macro** (confermato: sui small-tick fa ~0.32 = floor "predici-tutto-flat").
- **Metriche**: F1 macro, **MCC**, **p_T** (= probabilità di chiudere correttamente una transazione round-trip; metrica di trading). Miglior modello = max somma delle 3.
- Decision rule di Tabella 5: **non confermato** se argmax o con soglia (l'evaluator di LOBFrame calcola le metriche "as function of probability threshold", quindi potrebbe non essere argmax puro).

### CSCO @ orizzonte 50 — Tabella 5 (valori letti, ±0.02)

| Modello | F1 | MCC | p_T |
|---|---|---|---|
| **HLOB** (best) | **~0.60** | **~0.47** | ~0.09–0.16 |
| DeepLOB (2°) | ~0.58 | ~0.37 | ~0.13 |
| cluster cnn/transformer | 0.52–0.58 | 0.31–0.39 | — |
| iTransformer (worst) | 0.41 | 0.18 | 0.21 |

→ **L'intero campo di 10 modelli SOTA, con 3 anni di dati e split onesto, si ferma a F1 ~0.60 / MCC ~0.47 su CSCO.** Conferma esterna del soffitto informativo.

---

## 3. Tutti i nostri risultati

### 3a. Numeri consolidati (tuned, dove indicato)

| Setup | Split | F1 macro | MCC | Note |
|---|---|---|---|---|
| GCN — label **PCT** (θ=1e-4) | by_lag | **0.629** | — | **task vecchio, non tick**; acc 0.759; per-classe flat .84/down .54/up .50 |
| CGNN — tick | by_lag | 0.628 | — | argmax 0.522 → tuned 0.628 |
| STGCN — tick | by_lag | 0.623 | — | ≈ CGNN |
| CGNN + BiN — tick | by_lag | 0.569 | — | **BiN peggiora**, disattivato |
| **STHNN** lag100 — tick | by_lag | 0.6172 | — | esterno (RecurrentSparseSTHNN), label identiche |
| **STHNN** lag100 — tick | **by_file** | **0.6115** | — | split onesto |
| **STHNN** lag150 — tick | **by_file** | **0.6009** | — | split onesto (std); pen 0.5976 |

- **by_lag → by_file costa solo ~0.006 F1** (STHNN): il leak dello split ottimistico è minimo → coerente col soffitto (poco segnale da "rubare").
- STHNN by_file: Weighted F1 ~0.88, signal precision ~0.42–0.44, signal recall ~0.51–0.54.
- ⚠️ **GCN standalone su label TICK non è mai stato misurato** (baseline mancante). Il "0.629" è su label PCT (task diverso); il "0.628" è CGNN, non GCN nuda.

### 3b. Run nuovi SAGE / GAT / GCN (tick, `by_lag`) — SOLO traiettorie di val

Questi run hanno prodotto solo `metrics.csv` per-epoca (val, argmax). **I blocchi di test NON sono stati salvati** (run precedenti alla modifica di auto-save) → numeri di test persi se i terminali Lightning sono chiusi. Erano `by_lag` comunque, non i definitivi.

| Modello | best val_f1 | epoca | Osservazione |
|---|---|---|---|
| SAGE | 0.599 | 2 | il più alto |
| GAT | 0.557 | 2 | overfit più duro (val_loss esplode 0.58→0.86) |
| GCN | 0.549 | 7 | val_f1 molto rumoroso (0.42–0.55) |

**NON è un ranking**: val, argmax, `by_lag`, **singolo seed**, e "best" = max su epoche rumorose (biased verso l'alto). Il gap SAGE–GCN (0.05) sta dentro l'oscillazione epoca-epoca di GCN stesso (0.13). Tutti overfittano (train_acc → 0.72–0.74 mentre val_f1 resta ~0.50).

**Verdetto preliminare**: nessuna architettura vince — tutte nella banda ~0.55–0.60 val. L'attenzione di GAT **non aiuta** (anzi overfitta di più). Risposta alla domanda "SAGE/GAT per fare meglio?": **no**, non battono la GCN in modo reale.

---

## 4. Considerazioni metodologiche (importanti, evitano errori di valutazione)

1. **Soffitto informativo** — evidenza principale: (a) plateau di train_acc; (b) tutti i modelli/operatori di grafo ~0.60–0.63; (c) linear probe ≈ GNN; (d) il campo HLOB (10 SOTA, 3 anni) si ferma a 0.60. Il limite è il **segnale**, non l'architettura. NB: i confronti a singolo run entro ±0.01 sono **rumore**, non evidenza — il plateau lo è.

2. **Il tuning delle soglie NON gonfia** — è legittimo: leakage-free (tarato su val, applicato a test) ed è la scelta *corretta* per la macro-F1 su classi sbilanciate (l'argmax minimizza l'error-rate, non la macro-F1). LOBFrame stesso calcola metriche al variare della soglia → forse neanche HLOB è argmax puro. Va **tenuto e riportato**.

3. **`by_lag` è leaky ma poco** — `split_by_lag` (`src/dataset/preprocessing.py:63-102`) divide la timeline di **ogni giorno** in 70/15/15, quindi ogni giornata entra in train+val+test (leak intraday + nessun embargo a k=50). Empiricamente però costa solo ~0.006 F1 → non è il vero problema. Per il confronto con HLOB usare comunque **`by_file`** (onesto).

4. **MCC = il buco principale** — HLOB fa headline su F1 **+ MCC** (0.47); noi non l'abbiamo mai calcolato. Tarare le soglie per la macro-F1 **non** massimizza l'MCC → possiamo essere pari sulla F1 e indietro sull'MCC senza saperlo. **Ora aggiunto** (vedi §5).

5. **Dove siamo davvero** — su `by_file` + tuning la nostra famiglia (STHNN) sta a **~0.61 F1 ≈ HLOB 0.60**, con ~30× meno dati (5 giorni vs 3 anni). **Pari sulla F1, MCC ignoto.** Né dietro né davanti finché non abbiamo l'MCC.

6. **Confronti apples-to-apples** — non confrontare mai: 0.629 (PCT) con 0.60 (tick); 0.628 tuned by_lag con 0.60 argmax by_file; val con test. Tenere fisse: label, split, decision rule, metrica.

---

## 5. Stato del codice (modifiche fatte in questa sessione — LOCALI, da sincronizzare)

- **`src/training/metrics.py`**
  - `compute_metrics` ora ritorna anche **`"mcc"`** (`sklearn.matthews_corrcoef`).
  - Aggiunta `format_report()` (ritorna report+confusion come stringa).
- **`scripts/train.py`**
  - Stampa **MCC su argmax E su tuned** (per vedere se tarare per F1 costa MCC).
  - **Auto-save in `results/`**: per ogni run un `results/<model>_<split>_<timestamp>.txt` (report completo) + append di riga/e in **`results/summary.csv`** (colonne: `timestamp, model, split, rule, f1_macro, mcc, accuracy, f1_weighted, f1_down, f1_flat, f1_up, thr_down, thr_up, params`). Terminal-independent.
  - Nuovo flag **`--split {by_lag,by_file}`** (override di `data.split_strategy`; cambiare split ri-processa i tensori automaticamente).
- **`src/models/gat.py`** (sessione precedente): portato a **parità** con GCN/SAGE (mean+max pool, head `2*hidden`).
- **`config/default.yaml`**: preset `overrides` per sage/gat/cgnn/stgcn (~180–190k param).

> ⚠️ Tutte queste modifiche sono **sul repo locale**. Il training gira su **Lightning**: vanno portate lì (git push → pull) o **il prossimo run non avrà MCC né auto-save**. Non ancora committate.

---

## 6. Prossimi passi (per chiudere il confronto con HLOB)

1. **Sincronizzare il codice su Lightning** (commit + push + pull).
2. Lanciare il **trio onesto**, uno alla volta (il primo ri-processa a `by_file`; nessun run `by_lag` deve girare in parallelo — sovrascrive i tensori mmap):
   ```bash
   python scripts/train.py --model gcn  --split by_file   # baseline-tick MANCANTE
   python scripts/train.py --model sage --split by_file
   python scripts/train.py --model gat  --split by_file
   ```
3. Ognuno salva **F1 + MCC (argmax & tuned)** in `results/summary.csv`.
4. **≥3 seed** per modello (il val_f1 oscilla molto): serve la varianza, non un singolo picco. (Nota: `scripts/train.py` non ha `--seed`; il seed è in config `training.seed: 42` → editare o aggiungere flag.)
5. Piazzare i numeri (F1+MCC) dentro la riga CSCO di HLOB → verdetto onesto.

### Idee aperte (oltre il confronto)
- **Order-flow** (`src/dataset/order_flow.py`) come **feature di NODO** (non vettore globale a 6); aggiungere OFI, spread. Bloccato su disallineamento message/orderbook su 3/5 giorni.
- **Embargo/purge** su by_lag (López de Prado) — ma impatto atteso piccolo (~0.006).
- Metrica **p_T-like** (round-trip) per allinearsi alle 3 metriche di HLOB (abbiamo già signal precision/recall, flat-to-signal).

---

## 7. File e percorsi chiave

| Cosa | Percorso |
|---|---|
| Config centrale | `config/default.yaml` |
| Entry training | `scripts/train.py` |
| Threshold tuning standalone | `scripts/tune_threshold.py` |
| Metriche | `src/training/metrics.py` |
| Trainer | `src/training/trainer.py` |
| Split / preprocessing | `src/dataset/preprocessing.py` (`split_by_lag` :63, `split_by_file` :49) |
| Labeling | `src/dataset/labeling.py` |
| Modelli | `src/models/{gcn,cgnn,stgcn,sage,gat}.py` |
| Adiacenza TMFG | `../tmfg/cisco_tmfg_adj_matrix_2000_bins.csv` |
| Report STHNN (esterno) | `/Users/simone/Downloads/RecurrentSparseSTHNN Full Experiment Report.pdf` |
| Paper HLOB | `/Users/simone/Desktop/Info LOB.pdf` (arXiv:2405.18938) |
| Output run (nuovo) | `results/summary.csv` + `results/<model>_<split>_<ts>.txt` |

---

## 8. Riepilogo in una frase

> Sullo stesso task e split onesto, i nostri modelli GNN (~0.61 F1) sono **alla pari con HLOB (0.60)** su CSCO usando una frazione dei dati, ma manca l'**MCC** per dirlo con rigore; nessuna architettura (GCN/SAGE/GAT/CGNN/STGCN/STHNN) sfonda il **soffitto ~0.60**, che è un limite di **segnale**, non di modello.
