# HANDOFF — GNN su LOB CSCO vs benchmark HLOB

> Documento di passaggio (aggiornato **2026-09-06** dopo la campagna di 9 run: multi-seed, controllo SAGE a parametri appaiati, test di scala, griglia architettura×operatore). Autosufficiente: una chat nuova può ripartire da qui senza il transcript.
> Progetto: predizione direzione mid-price (down/flat/up) da LOB CSCO (LOBSTER) con GNN su grafo TMFG.

---

## TL;DR

- Il nostro task ha **le stesse identiche label** del paper **HLOB** (Briola–Bartolucci–Aste): differenza mid-price punto-a-punto, soglia 1 tick, orizzonte 50 (verificato su Eq. 2 del PDF). Quindi il confronto è **legittimo**.
- **HLOB su CSCO @ orizzonte 50: F1 ≈ 0.60, MCC ≈ 0.40, p_T ≈ 0.16** (il migliore di 10 modelli SOTA; l'intero campo si ferma lì). ⚠️ La prima versione di questo documento riportava **MCC 0.47: era la riga BAC** della Tabella 5, letta per errore (HLOB su BAC: 0.62/0.47/0.09).
- **Tutte e 9 le configurazioni misurate su `by_file`** (`results/summary.csv`). Migliore assoluto: **CGNN scalato a 720k param → F1 0.6428 / MCC 0.4483**. A budget ~180k: STGCN 0.6282±0.0076, CGNN 0.6259±0.0031 (3 seed). Tutti sopra HLOB (0.60/0.40).
- **Il rumore da seed è piccolo: σ ≈ 0.003–0.008.** Quindi differenze ≥0.015 sono reali; CGNN e STGCN a budget sono indistinguibili tra loro. Il sorpasso su HLOB (~0.03) sopravvive al multi-seed.
- ⚠️ **IPOTESI FALSIFICATA — il "soffitto ~0.62" era in parte capacità, non informazione.** Il test di scala: CGNN 181k→720k dà 0.626→**0.643** (+0.016, ~5σ); SAGE 182k→272k dà 0.609→0.625 (+0.017). Le versioni precedenti di questo doc affermavano che più parametri non avrebbero aiutato: **sbagliato**. Dove satura la scala è ora la domanda aperta principale.
- **La CNN temporale aiuta la GCN, NON SAGE — chiuso con controllo a parametri appaiati.** ~180k: SAGE 0.609 > CNN+SAGE 0.601. ~270k: SAGE 0.625 > CNN+SAGE 0.615. A entrambi i budget SAGE puro vince → il 0.615 era capacità, non la CNN.
- **GAT è il peggior operatore in tutte e 3 le famiglie** (statica 0.584, CGNN 0.606, STGCN 0.598): il TMFG ha già filtrato gli archi, l'attention aggiunge varianza senza selettività.
- ⚠️ **Due sottocampioni di test**: i run di luglio (support down/up 3219/2925) e quelli di settembre (3120/3009) usano 50k diversi dello stesso giorno. I 9 run di settembre condividono lo stesso test set (confronti puliti); le medie 3-seed mescolano i due. Effetto ~±0.005, non ribalta nulla. **Causa rimossa**: il sottocampione ora usa `data.subsample_seed` fisso, scollegato da `training.seed`.

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
- **Modelli** (`src/models/`): 3 famiglie × 3 operatori. Statiche: `gcn`, `sage`, `gat`. Spazio-temporali: `cgnn` / `cgnn_sage` / `cgnn_gat` (CNN sui lag → grafo) e `stgcn` / `stgcn_sage` / `stgcn_gat` (blocchi intercalati). L'operatore si sceglie con `conv_type` via `make_graph_conv()` in `base.py`. Tutti mean+max pool, LayerNorm, lag feature, ~180–206k param. Le varianti **GAT girano sul path PyG** (non static).
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
- **Protocollo**: training bilanciato (5k/classe/giorno); **test = distribuzione naturale su 10 giorni held-out** (≈ nostro `by_file`; NB: i loro 5 giorni di val sono estratti *dentro* il periodo di training, non dopo); metriche **mediate su 3 anni** (2017–2019, 40 train + 5 val + 10 test giorni/anno). **F1 = macro per inferenza, non dichiarato**: la parola "macro" non compare nel paper, ma sui small-tick HLOB fa ~0.32 = esattamente il floor macro-F1 di "predici-tutto-flat" (≈0.316), quindi l'inferenza è solida.
- **Metriche**: F1 macro, **MCC**, **p_T** (= probabilità di chiudere correttamente una transazione round-trip; metrica di trading). Miglior modello = max somma delle 3.
- Decision rule di Tabella 5: **non confermato** se argmax o con soglia (l'evaluator di LOBFrame calcola le metriche "as function of probability threshold", quindi potrebbe non essere argmax puro).

### CSCO @ orizzonte 50 — Tabella 5 (valori VERIFICATI via estrazione testo dal PDF)

| Modello | F1 | MCC | p_T |
|---|---|---|---|
| **HLOB** (best) | **0.60** | **0.40** | 0.16 |
| DeepLOB (2°) | 0.58 | 0.37 | 0.13 |
| cluster cnn/transformer/tabl | 0.52–0.57 | 0.31–0.36 | 0.10–0.15 |
| iTransformer (worst) | 0.41 | 0.18 | 0.11 |

> ⚠️ **Correzione (2026-07-08)**: la prima versione riportava HLOB MCC ~0.47 e p_T ~0.09–0.16 — erano valori della riga **BAC** (HLOB su BAC: 0.62/0.47/0.09). La riga CSCO corretta è **0.60/0.40/0.16**. Anche il p_T di iTransformer era errato (0.21 → 0.11). F1/MCC di DeepLOB e del cluster erano corretti.

→ **L'intero campo di 10 modelli SOTA, con 3 anni di dati e split onesto, si ferma a F1 0.60 / MCC 0.40 su CSCO.** NB: è il limite *raggiunto da loro*, non un soffitto dimostrato del task — i nostri modelli lo superano (§3a).

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
| **CGNN scalato** (h350/cnn128, 720k) | **by_file** | **0.6428** | **0.4483** | ᴮ **BEST ASSOLUTO**; val_f1=0.641; +0.016 (~5σ) sul CGNN a budget → la scala non è satura |
| **STGCN** (183k) — 3 seed | **by_file** | **0.6282 ± 0.0076** | **0.4286 ± 0.0103** | seed 42ᴬ 0.6195 / 43ᴮ 0.6315 / 44ᴮ 0.6337 |
| **CGNN** (181k) — 3 seed | **by_file** | **0.6259 ± 0.0031** | **0.4235 ± 0.0045** | seed 42ᴬ 0.6246 / 43ᴮ 0.6237 / 44ᴮ 0.6295 |
| **STGCN-SAGE** (182k) | **by_file** | 0.6324 | 0.4322 | ᴮ miglior singolo a budget, ma dentro il rumore di STGCN |
| **SAGE @272k** (h196) | **by_file** | 0.6253 | 0.4204 | ᴮ **controllo a param appaiati**: batte CNN+SAGE@273k (0.6153) → la CNN non aiuta SAGE |
| **SAGE** (182k) | **by_file** | 0.6086 | 0.3976 | ᴬ best GNN statica a budget |
| **CGNN-GAT** (186k) | **by_file** | 0.6057 | 0.3978 | ᴮ GAT peggiora anche dentro CGNN (−0.02 vs cgnn) |
| **CGNN-SAGE** (h175, 273k) | **by_file** | 0.6153 | 0.4078 | ᴬ fuori budget; battuto da SAGE puro a pari param |
| **CGNN-SAGE** (h140, budget) | **by_file** | 0.6008 | 0.3868 | ᴬ a budget −0.008 vs SAGE |
| **STGCN-GAT** (186k) | **by_file** | 0.5975 | 0.3794 | ᴮ peggiore della griglia spazio-temporale |
| **GAT** (188k) | **by_file** | 0.5841 | 0.3710 | ᴬ il tuning qui *peggiora* (argmax 0.5874) |
| **GCN** (206k) | **by_file** | 0.5715 | 0.3427 | ᴬ peggiore da sola; +0.053 con la CNN (→CGNN) |

> ᴬ = test subsample di luglio (support down/up 3219/2925) · ᴮ = test subsample di settembre (3120/3009). Confronti **puliti solo entro lo stesso gruppo**; vedi TL;DR.

- **Griglia architettura × operatore** (~180k param, F1 tuned):

  | | GCN | SAGE | GAT |
  |---|---|---|---|
  | statica | 0.572 | 0.609 | 0.584 |
  | CGNN | **0.626** | 0.601 | 0.606 |
  | STGCN | **0.628** | **0.632** | 0.598 |

  Le righe spazio-temporali stanno sopra la statica (tranne con GAT); la colonna GAT è la peggiore ovunque.
- **Test di scala**: SAGE 182k→272k = 0.609→0.625; CGNN 181k→720k = 0.626→**0.643**. Entrambi ~+0.017 → **capacità non satura**.
- **by_lag → by_file trascurabile** (~0.004–0.006): confermato su STHNN, CGNN, STGCN.
- Il tuning ha generalizzato ovunque (val_f1 → test entro ~0.01). Eccezione GAT statica, dove peggiora.
- Il "0.629" storico è su label PCT (task diverso), non confrontabile.

### 3b. Run nuovi SAGE / GAT / GCN (tick, `by_lag`) — SOLO traiettorie di val

Questi run hanno prodotto solo `metrics.csv` per-epoca (val, argmax). **I blocchi di test NON sono stati salvati** (run precedenti alla modifica di auto-save) → numeri di test persi se i terminali Lightning sono chiusi. Erano `by_lag` comunque, non i definitivi. **(Ora superato: il trio è stato rieseguito su `by_file` con auto-save — vedi §3a.)**

| Modello | best val_f1 | epoca | Osservazione |
|---|---|---|---|
| SAGE | 0.599 | 2 | il più alto |
| GAT | 0.557 | 2 | overfit più duro (val_loss esplode 0.58→0.86) |
| GCN | 0.549 | 7 | val_f1 molto rumoroso (0.42–0.55) |

**NON è un ranking**: val, argmax, `by_lag`, **singolo seed**, e "best" = max su epoche rumorose (biased verso l'alto). Il gap SAGE–GCN (0.05) sta dentro l'oscillazione epoca-epoca di GCN stesso (0.13). Tutti overfittano (train_acc → 0.72–0.74 mentre val_f1 resta ~0.50).

**Verdetto — SUPERATO dai run `by_file` di §3a**: questi run di val/by_lag non predicevano il ranking finale. Il ranking onesto a budget (F1 tuned) è **STGCN-SAGE 0.632 ≈ STGCN 0.628 ≈ CGNN 0.626 > SAGE 0.609 > CGNN-GAT 0.606 > CGNN-SAGE 0.601 > STGCN-GAT 0.598 > GAT 0.584 > GCN 0.572**, e fuori budget CGNN@720k 0.643. Decide la **dimensione temporale**, non l'operatore; GAT è sempre in coda.

---

## 4. Considerazioni metodologiche (importanti, evitano errori di valutazione)

1. ⚠️ **Il "soffitto informativo ~0.62" è FALSIFICATO** — le versioni precedenti di questo doc lo davano per assodato ("più parametri non aiutano"). Il test di scala dice il contrario: CGNN 181k→720k = 0.626→**0.643**, SAGE 182k→272k = 0.609→0.625, entrambi ~+0.017 contro un rumore da seed di ±0.003–0.008. Era **almeno in parte un limite di capacità**. Resta vero che il campo HLOB (10 SOTA, 3 anni) si ferma a 0.60, ma quello è il loro limite, non necessariamente il nostro. **Dove satura la scala: ignoto** — domanda aperta n.1.

2. **Il tuning delle soglie NON gonfia** — è legittimo: leakage-free (tarato su val, applicato a test) ed è la scelta *corretta* per la macro-F1 su classi sbilanciate (l'argmax minimizza l'error-rate, non la macro-F1). LOBFrame stesso calcola metriche al variare della soglia → forse neanche HLOB è argmax puro. Va **tenuto e riportato**.

3. **`by_lag` è leaky ma poco** — `split_by_lag` (`src/dataset/preprocessing.py:63-102`) divide la timeline di **ogni giorno** in 70/15/15, quindi ogni giornata entra in train+val+test (leak intraday + nessun embargo a k=50). Empiricamente costa solo ~0.004–0.006 F1 (ora confermato su STHNN, CGNN e STGCN) → non è il vero problema. Per il confronto con HLOB usare comunque **`by_file`** (onesto).

4. **MCC — sopra HLOB, confermato multi-seed** — su CSCO HLOB fa 0.40 (non 0.47: quello era BAC). Noi: CGNN@720k **0.4483**, STGCN 0.4286±0.0103, CGNN 0.4235±0.0045. Il timore "tarare per F1 costa MCC" non si è avverato: i leader migliorano anche l'MCC dopo tuning. Unica eccezione GAT statica, dove il tuning peggiora entrambe.

5. **Dove siamo davvero** — i migliori **superano HLOB (0.60/0.40)** su entrambe le metriche di ~0.03–0.04, con ~30× meno dati, e il margine regge al multi-seed (>3σ). **Ma** tutto vive su **un solo giorno di test** (5gg ⇒ 1 held-out) contro la media HLOB su 10gg × 3 anni: il multi-seed copre la varianza di inizializzazione, **non** quella giorno-per-giorno, che resta ignota ed è il limite più serio. Formulazione onesta: *sopra HLOB su questo titolo e questo giorno*.

6. **Confronti apples-to-apples** — non confrontare mai: 0.629 (PCT) con 0.60 (tick); 0.628 tuned by_lag con 0.60 argmax by_file; val con test. Tenere fisse: label, split, decision rule, metrica.

---

## 5. Stato del codice (tutto COMMITTATO)

- **`src/training/metrics.py`**: `compute_metrics` ritorna anche **`mcc`**; `format_report()` restituisce report+confusion come stringa.
- **`scripts/train.py`**: MCC su argmax **e** tuned; **auto-save** in `results/<model>_<split>_<ts>.txt` + append su `results/summary.csv`; flag **`--split`**, **`--seed`**, **`--hidden`**, **`--cnn-channels`** (quest'ultimo per i test di scala sulla CNN temporale).
- **`src/models/base.py`**: `make_graph_conv(conv_type, ...)` — factory unica per `gcn|sage|gat`, usata da CGNN e STGCN. GAT con `concat=False` per non alterare le dimensioni di residui/LayerNorm.
- **`src/models/{cgnn,stgcn}.py`**: parametri `conv_type` e `num_heads` → 6 varianti spazio-temporali registrate in `__init__.py` (`cgnn`, `cgnn_sage`, `cgnn_gat`, `stgcn`, `stgcn_sage`, `stgcn_gat`).
- **`src/dataset/preprocessing.py`**: ⚠️ fix importante — il sottocampione val/test usa ora **`data.subsample_seed` (default 42)** invece di `training.seed`. Prima, in una sessione con cache vuota, il primo run decideva il test set: è così che sono nati i due sottocampioni ᴬ/ᴮ di §3a.
- **`config/default.yaml`**: preset per tutte e 9 le configurazioni (~180–206k param).
- **`environment.yml`**: aggiunto **`torch_geometric`** (mancava: ogni studio Lightning nuovo si rompeva all'import).

> Nota operativa: le varianti **GAT non usano lo static path** (l'attention non è affidabile col trucco `[B,N,C]` a grafo condiviso) → girano su PyG, più lente. Vedi la lista in `train.py` (`use_static_mode`).

---

## 6. Prossimi passi

Tutti i passi della campagna precedente sono **completati** (9 run, `results/summary.csv`): multi-seed su CGNN/STGCN, controllo SAGE a parametri appaiati, test di scala, griglia 2×3 architettura×operatore. Restano:

1. **Fin dove scala?** — domanda aperta principale, nata dal fatto che 720k > 181k. Prossimo punto: `--model cgnn --split by_file --hidden 500 --cnn-channels 192` (~1.4M param). Se sale ancora, il limite non è l'informazione; se satura, hai finalmente localizzato il soffitto vero.
2. **Multi-seed sul CGNN@720k** (`--seed 43/44`): il 0.6428 è il numero di punta ed è su singolo seed.
3. **Più giorni di dati** — l'unica via per stimare la varianza giorno-per-giorno e rendere il confronto con HLOB pienamente equo.
4. **Metrica p_T-like** (round-trip) per allinearsi alle 3 metriche di HLOB.

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
| Modelli | `src/models/{gcn,cgnn,stgcn,sage,gat}.py` (CGNN-SAGE = CGNN con `conv_type="sage"`, registrato come `cgnn_sage` in `__init__.py`) |
| Adiacenza TMFG | `../tmfg/cisco_tmfg_adj_matrix_2000_bins.csv` |
| Report STHNN (esterno) | `/Users/simone/Downloads/RecurrentSparseSTHNN Full Experiment Report.pdf` |
| Paper HLOB | `/Users/simone/Desktop/Info LOB.pdf` (arXiv:2405.18938) |
| Output run (nuovo) | `results/summary.csv` + `results/<model>_<split>_<ts>.txt` |

---

## 8. Riepilogo in una frase

> Sullo stesso task e split onesto, i nostri modelli **spazio-temporali** superano HLOB su CSCO (0.60 F1 / 0.40 MCC) con ~30× meno dati: **CGNN scalato a 720k fa 0.643 / 0.448**, e a budget ~180k STGCN e CGNN stanno a 0.628±0.008 e 0.626±0.003 — margine >3σ, confermato multi-seed. L'ingrediente decisivo è la **dimensione temporale** (la CNN sui lag aiuta la GCN, non SAGE, verificato a parametri appaiati); l'**attention GAT è sempre la scelta peggiore**; e il presunto **soffitto ~0.62 è falsificato**: scalare i parametri continua a pagare, e dove saturi è la domanda aperta. Caveat da non nascondere: **un solo giorno di test**.
