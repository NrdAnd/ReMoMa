# Report — GNN per la predizione di direzione del prezzo da Limit Order Book

**Progetto:** predizione della direzione del mid-price (down / flat / up) da dati LOB (CSCO, LOBSTER) tramite Graph Neural Network su grafo TMFG.

Questo documento riassume tutto il lavoro svolto: decisioni di design, bug trovati e risolti, esperimenti, risultati e conclusioni.

---

## 1. Setup del problema

- **Dato grezzo:** 5 giorni di order book CSCO (LOBSTER, gennaio 2019), formato 40 colonne (10 livelli × {AskPrice, AskVol, BidPrice, BidVol}), ~850K tick/giorno.
- **Grafo:** TMFG (Triangulated Maximally Filtered Graph) → **3020 nodi** = 2 lati × 10 livelli × 151 lag, **9054 archi non orientati** (18108 diretti).
  - Ordinamento nodi: `ask_i_lag_k → i*151+k`, `bid_i_lag_k → 1510+i*151+k`.
- **Feature per nodo:** `[prezzo normalizzato, volume binnato]` (2 feature), poi diventate 3 con l'aggiunta del lag.
- **Task:** classificazione ternaria della direzione del mid-price a orizzonte `k` tick.
- **Metrica:** **F1 macro** (obbligatoria per classi sbilanciate — l'accuracy è ingannevole quando il flat domina).
- **Pipeline:** preprocessing offline → tensori `.npy` mmap → training PyTorch Geometric. Split temporale `by_lag` (70/15/15 dentro ogni file), train bilanciato a 50k campioni/classe.
- **Budget modello:** ~180K parametri.

---

## 2. Scelta dell'architettura: GCN

Prima architettura scelta: **GCN**, per tre motivi:
1. **Velocità** — GCN è 3-5× più veloce di GAT (niente attention per arco).
2. **Budget 180K** — con `hidden=225, layers=3` il GCN ha ~180K parametri; SAGE con gli stessi iperparametri arriverebbe a ~333K.
3. **TMFG già filtrato** — il grafo ha già selezionato gli archi informativi, quindi l'aggregazione uniforme di GCN è adeguata (non serve l'attention per ignorare archi inutili).

Aggiunta di un **node encoder** `Linear(2→225)` prima del message passing, calcolato per centrare ~180K parametri.

---

## 3. Il primo grande bug: feature volume azzerata

**Sintomo:** loss bloccata a `log(3) ≈ 1.099`, output costante (una sola classe), su *ogni* configurazione provata.

**Diagnosi (con script dedicati `diagnose.py`, `inspect_volume.py`):**
- Test di overfit su un batch piccolo → il modello non fittava nemmeno i dati visti.
- Statistiche feature → **il canale volume era identicamente zero**.
- Causa radice: poche righe dei CSV grezzi contenevano valori **NaN** nei volumi. `np.percentile` propaga il NaN → `np.unique` collassa gli edge del binner a `[nan]` → il `VolumeBinner` restituiva 0 per ogni volume.

**Fix (`binning.py`):** filtro `np.isfinite` nel `fit()`, `nan_to_num` nel transform, e una **guardia** che solleva un errore se gli edge collassano (così un bug simile non passerà mai più in silenzio).

> Lezione: un singolo NaN in input azzerava silenziosamente metà delle feature. La guardia difensiva è ora parte del codice.

---

## 4. Catena di miglioramenti (etichette percentuali, k=50)

Risolto il bug, una serie di interventi mirati ha alzato progressivamente l'F1 sul test:

| Intervento | Test F1 macro |
|---|---|
| Bug volume risolto (baseline) | 0.534 |
| + feature **lag** (positional encoding) + **LayerNorm** + **lr 3e-4** | 0.610 |
| + **mean+max pooling** | **0.629** |

Dettagli chiave:
- **Feature di lag:** ogni nodo riceve `lag/n_lags ∈ [0,1]`, aggiunta nel modello come buffer (non su disco). Dà al modello la posizione temporale del nodo. +225 parametri.
- **BatchNorm → LayerNorm:** la BatchNorm causava una grave **instabilità train/eval** (train liscio, val che oscillava tra 0.60 e 0.23). Causa: le *running statistics* della BatchNorm inseguono in ritardo un modello che cambia in fretta, e in `eval()` producono predizioni sballate. La LayerNorm non ha statistiche correnti → comportamento identico in train e eval → curva di validazione liscia. **Fix fondamentale.**
- **Learning rate:** tarato con un *LR range test*. 3e-4 (con LayerNorm stabile) converge veloce; 2e-4 più conservativo.
- **mean+max pooling:** invece del solo `global_mean_pool` (che media via i picchi su 3020 nodi), concatena anche il `global_max_pool` → preserva il nodo più saliente. Guadagno reale sulle classi difficili.

---

## 5. La scoperta del "tetto statico" (~0.63)

Esaurita la catena di miglioramenti, abbiamo indagato il limite:

- **GCN ≈ GAT ≈ SAGE** — cambiare la variante di GNN **non** sposta il risultato (~0.63). Tutte aggregano in modo diverso ma estraggono lo stesso segnale.
- **Un probe lineare (regressione logistica su poche feature a mano) eguaglia la GNN** — il segnale predittivo è essenzialmente l'**order-book imbalance** (sbilanciamento di volume bid/ask al livello 1), che è **linearmente separabile**.
- **Il train accuracy si ferma a ~0.71** su ogni architettura → il modello non riesce a fittare meglio nemmeno i dati di training → **limite informativo**, non di ottimizzazione né di architettura.

> Conclusione: i modelli **statici** (che vedono lo *snapshot* del book) saturano a ~0.63 perché tutto il segnale separabile è nell'imbalance, già catturato anche da un modello lineare.

Strumenti di verifica creati: `diagnose.py` (overfit + equivalenza forward), `verify_train.py` (bilanciamento classi + probe lineare), `lr_finder.py`, `inspect_volume.py`.

---

## 6. La svolta temporale: rompere il tetto statico

**Intuizione:** i nodi-lag contengono la dimensione temporale, ma GCN+pooling la **appiattisce** (message passing cieco all'ordine + pooling globale che media via il tempo). Sapere la *posizione* di un nodo (feature lag) ≠ modellare la *dinamica* (come cambiano i valori nel tempo).

**Modelli spazio-temporali (GNN-based, niente LSTM):**

- **CGNN** (`cgnn.py`): una **CNN 1D sui lag** (per ogni gruppo lato/livello) estrae i pattern temporali → poi GNN sul grafo TMFG → mean+max pool.
  - **Test F1 0.651** (vs 0.629 statici) — **primo modello a rompere il tetto statico.**
  - Train acc salito a 0.76 (oltre il muro di 0.71) → estrae segnale temporale nuovo.
- **STGCN** (`stgcn.py`): blocchi `[temporale → grafo → temporale]` alternati e ripetuti sullo stesso grafo.
  - **≈ CGNN** (non meglio) → anche le varianti temporali convergono.

> Conclusione: la dimensione temporale aggiunge segnale reale (CGNN > statici), ma **più sofisticazione architetturale non aiuta oltre** (STGCN = CGNN).

---

## 7. Etichettatura in tick (economicamente motivata)

Siamo passati da una **soglia relativa** (es. 1e-4) a una **soglia assoluta di un tick** ($0.01).

**Dettaglio critico:** i prezzi LOBSTER sono in unità di `$ × 10000`, quindi un tick ($0.01) = **soglia 100** (non 0.01). Errore inizialmente fatto e poi corretto.

**Motivazione (microstruttura + costi di esecuzione):** un movimento sotto un tick non è *tradeable* — viene mangiato dallo spread bid-ask e dalle commissioni. Etichettare come up/down solo i movimenti ≥ 1 tick allinea il target a ciò che è realmente sfruttabile. È la direzione **"Average Spread Thresholding"** che la letteratura più avanzata (TLOB) indica come corretta.

**Conseguenza:** task molto più **sbilanciato (~90% flat, ~5% down, ~5% up)**, perché CSCO è molto liquido e il mid si muove di un tick intero raramente su k=50.

---

## 8. Gestione dello sbilanciamento: threshold tuning

Su etichette in tick, il modello (allenato bilanciato 33/33/33) **sovra-predice** le classi rare quando valutato sulla distribuzione reale (90% flat): recall alto ma **precision bassa** su down/up → F1 macro basso.

**Soluzione (`threshold.py`, `tune_threshold.py`):** **threshold tuning** post-hoc. Si tiene il modello fisso e si cambia solo la *regola di decisione* — predire down/up solo se la probabilità supera una soglia tarata. Le soglie si ottimizzano sul **val** (massimizzando l'F1 macro) e si applicano al **test** (metodologicamente corretto, niente leakage).

| | argmax | threshold-tuned |
|---|---|---|
| **CGNN** (etichette tick) | 0.522 | **0.628** (+0.106) |

Il guadagno è gratuito (nessun riaddestramento). Integrato anche in `train.py` (flag `tune_threshold: true`).

---

## 9. Cosa NON ha funzionato (risultati negativi importanti)

| Tentativo | Esito |
|---|---|
| Più dati (300k train) | nessun guadagno |
| SAGE / STGCN vs CGNN | ≈ uguali (architettura satura) |
| **BiN** (Bilinear Normalization, dalla letteratura SOTA) | **peggiora** (0.628 → 0.569) |

In tutti questi casi il **train acc resta ~0.76** → conferma che siamo al **tetto informativo** degli orderbook snapshot con etichette in tick. Tutto ciò che **non aggiunge informazione** (architettura, dati, normalizzazione) converge allo stesso punto.

---

## 10. Posizionamento rispetto alla letteratura

Abbiamo analizzato un survey completo sullo stato dell'arte (CNN → LSTM → DeepLOB → TABL → Transformer/TLOB). Punti chiave:

- **Le GNN/grafi sono ASSENTI dal survey** → il nostro approccio è una **novità** rispetto al mainstream (tutto CNN/LSTM/Transformer).
- I punteggi alti della letteratura (DeepLOB ~80%, TLOB ~92.8%) sono su **FI-2010** con **labeling smoothed** (più facile, più bilanciato) — **NON confrontabili** col nostro task in tick (più duro, onesto). Il survey stesso ammette che con **etichette economicamente oneste** (spread-based) i punteggi dei Transformer **crollano**.
- Il survey conferma le nostre scoperte: il caso **MLPLOB** (un MLP banale + normalizzazione batte modelli complessi) → **"il collo di bottiglia è il pre-processing, non l'architettura"** — esattamente ciò che abbiamo visto.
- Tema dominante: **simulation-to-reality gap** ed efficienza dei mercati (EMH) → i modelli da laboratorio non generalizzano e non fanno profitto dopo i costi.

> Il nostro lavoro è metodologicamente **rigoroso**: etichette oneste, GNN come approccio novel, risultato coerente con la teoria (segnale informazionalmente limitato).

---

## 11. La leva attuale: Order Flow dai message file

Esaurite le leve che riorganizzano lo stesso segnale, l'unica che aggiunge **informazione nuova** è l'**order flow** dai message file di LOBSTER (i trade eseguiti e gli ordini aggressivi, che **non** sono negli snapshot del book).

**Validato (`order_flow.py`):** il volume firmato dei trade aggressivi negli ultimi 50 eventi è **predittivo** della direzione futura:

```
direzione futura (k=50)   trade-flow recente (media)
  down                      −347   (vendite aggressive)
  flat                       −74
  up                        +104   (acquisti aggressivi)
```

Relazione monotòna → segnale direzionale reale, e nuovo (non nello snapshot).

**Problema dati riscontrato:** i file orderbook e message **non sono allineati** su 3 giorni su 5 (gli orderbook hanno perso righe — i message sono più completi). Servono coppie coerenti (re-download) per integrare l'order flow su tutti i giorni.

| Giorno | message | orderbook | allineati |
|---|---|---|---|
| 2019-01-22 | 961.786 | 849.735 | ✗ |
| 2019-01-23 | 933.023 | 858.811 | ✗ |
| 2019-01-24 | 828.838 | 828.838 | ✓ |
| 2019-01-25 | 700.260 | 678.638 | ✗ |
| 2019-01-28 | 828.085 | 828.085 | ✓ |

---

## 12. Risultato attuale e stato del progetto

- **Miglior modello: CGNN (CNN temporale + GNN) — F1 macro ~0.628 (threshold-tuned)** su etichette in tick, eval sulla distribuzione reale (90% flat).
- Progressione complessiva:
  ```
  0.534  bug volume risolto
  0.610  lag + LayerNorm + lr 3e-4
  0.629  + mean+max pooling      (tetto statico)
  0.651  CGNN temporale          (etichette pct — rompe il tetto statico)
  0.628  CGNN + threshold tuning (etichette tick, task più onesto e duro)
  ```
- **Conclusione scientifica:** su etichette economicamente oneste, un approccio GNN (novità) plateau a ~0.63 perché il segnale degli snapshot è **informazionalmente limitato** — coerente col messaggio EMH/sim-to-real del survey. I risultati negativi (architettura, dati, BiN tutti saturi) sono parte del contributo.

---

## 13. Direzioni aperte

1. **Order Flow (in corso)** — integrare le feature dai message file (segnale nuovo validato). Richiede coppie orderbook/message allineate (re-download dei 3 giorni corrotti).
2. **Classificazione a due stadi** — "movimento vs fermo" poi "up vs down" (gestione sbilanciamento).
3. **Smoothed labeling** (FI-2010) — ridurre il rumore delle etichette (media dei k futuri invece del singolo prezzo).
4. **Attention pooling** — pesi imparati sui nodi invece di mean+max.

---

## 14. File principali del progetto

| File | Ruolo |
|---|---|
| `config/default.yaml` | configurazione (modello, split, etichette, training) con preset per architettura |
| `scripts/train.py` | training end-to-end (override CLI `--model`, threshold tuning integrato) |
| `scripts/preprocess_dataset.py` | rigenerazione tensori processati |
| `scripts/tune_threshold.py` | threshold tuning post-hoc su un checkpoint |
| `scripts/diagnose.py`, `verify_train.py`, `lr_finder.py`, `inspect_volume.py` | diagnostica |
| `src/models/{gcn,gat,sage,cgnn,stgcn}.py` | architetture (statiche + spazio-temporali) |
| `src/models/bin.py` | Bilinear Normalization (provata, disattivata) |
| `src/dataset/{binning,labeling,preprocessing,order_flow}.py` | pipeline dati |
| `src/training/{trainer,metrics,threshold}.py` | training, metriche, threshold tuning |

**Come allenare:**
```bash
python scripts/train.py --model cgnn       # CGNN (miglior modello)
python scripts/tune_threshold.py --model cgnn   # taratura soglie su checkpoint
```
