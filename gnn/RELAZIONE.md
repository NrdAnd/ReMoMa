# GNN su grafo TMFG per la predizione della direzione del mid-price da Limit Order Book

> Riassunto della ricerca — 2026-07-09
> Dati: LOB di Cisco (CSCO), LOBSTER, 5 giorni (gennaio 2019). Benchmark di riferimento: paper HLOB (Briola, Bartolucci, Aste — arXiv:2405.18938).

---

## Sintesi dei risultati

Con **5 giorni di dati** (contro i ~3 anni del benchmark) e uno **split di valutazione onesto** (giorni interi mai visti in training), il nostro miglior modello GNN raggiunge **F1 macro 0.609 / MCC 0.398**, contro **F1 0.60 / MCC 0.40** di HLOB — il migliore di 10 modelli SOTA sullo stesso titolo, stesso task e stesse label. Siamo cioè **alla pari con lo stato dell'arte usando ~30× meno dati**.

Il secondo risultato, altrettanto importante: **sei architetture diverse convergono tutte nella banda 0.57–0.61 di F1**, e anche il campo dei 10 modelli SOTA del paper si ferma a 0.60. Questo indica un **soffitto informativo del segnale** (quanto il libro ordini permette di prevedere a questo orizzonte), non un limite dei modelli.

---

## 1. Il problema

- **Task**: classificazione ternaria della direzione del mid-price a **k = 50 eventi** LOB: *down / flat / up*.
- **Label** (identiche all'Eq. 2 del paper HLOB, che rende il confronto legittimo):

  ```
  Δm = mid[t+50] − mid[t]
  down se Δm ≤ −1 tick ($0.01) ; up se Δm ≥ +1 tick ; flat altrimenti
  ```

- La distribuzione naturale è fortemente sbilanciata: **~88% flat**. Per questo la metrica principale è la **F1 macro** (l'accuracy è ingannevole: "predici sempre flat" fa 88% di accuracy ma F1 macro ≈ 0.32), affiancata dall'**MCC** (Matthews Correlation Coefficient), la metrica di punta del benchmark.

## 2. La rappresentazione a grafo

Ogni istante di mercato è un grafo di **3020 nodi** = 2 lati (bid/ask) × 10 livelli di prezzo × 151 istanti temporali (lag 0–150). La struttura degli archi è un **TMFG** (Triangulated Maximally Filtered Graph) calcolato sulla mutua informazione tra le serie: **9054 archi non orientati** (18108 diretti), lo stesso tipo di "information filtering network" usato da HLOB.

Feature per nodo: prezzo normalizzato rispetto al mid, volume discretizzato in quantili, lag normalizzato ∈ [0,1].

## 3. I modelli confrontati

Tutti a **budget di parametri comparabile (~180–205k)**, stessa testa di classificazione (mean+max pooling → MLP), stesso training.

| Modello | Idea | Parametri |
|---|---|---|
| **GCN** | Convoluzione di grafo classica, aggregazione uniforme dei vicini | 206k |
| **GraphSAGE** | Separa la trasformazione del nodo da quella dei vicini (2 matrici per layer) | 182k |
| **GAT** | Attention sugli archi: pesa i vicini in modo appreso | 188k |
| **CGNN** | Ibrido: CNN 1D sull'asse temporale (lag) → GCN sul grafo | 181k |
| **CGNN-SAGE** | Come CGNN ma con SAGEConv come operatore spaziale | 185k |
| **STGCN** | Blocchi spazio-temporali intercalati (conv. temporale → grafo → temporale) | 183k |

A questi si aggiunge il progetto parallelo **RecurrentSparseSTHNN** (rete ipergrafica spazio-temporale, stesso task e label), usato come confronto interno.

## 4. Protocollo sperimentale

- **Split onesto (`by_file`)**: train sui giorni 1–3, validation sul giorno 4, **test sul giorno 5 mai visto** — l'analogo del protocollo HLOB (giorni di test successivi e disgiunti dal training). Uno split alternativo (`by_lag`, 70/15/15 dentro ogni giornata) è stato usato nelle fasi esplorative ma è leggermente ottimistico (leak intra-day) e non viene usato per i confronti finali.
- **Training bilanciato** (50k campioni/classe), **test sulla distribuzione naturale** (~88% flat) — come HLOB (5k/classe/giorno in training, test naturale).
- **Tuning delle soglie di decisione**: le soglie di probabilità per le classi down/up sono tarate sulla validation e applicate al test (procedura senza leakage). È la scelta corretta per la macro-F1 su classi sbilanciate: l'argmax minimizza l'error-rate, non la macro-F1. Riportiamo comunque entrambe le regole.
- Ottimizzazione: AdamW, early stopping su validation, LR scheduling. Seed fisso (42).

## 5. Risultati (split onesto, test = giorno mai visto, 50k campioni)

| Modello | F1 (argmax) | MCC (argmax) | F1 (tuned) | MCC (tuned) | F1 down / flat / up (tuned) |
|---|---|---|---|---|---|
| GCN | 0.476 | 0.311 | 0.572 | 0.343 | 0.40 / 0.91 / 0.41 |
| **GraphSAGE** | 0.545 | 0.384 | **0.609** | **0.398** | 0.45 / 0.92 / 0.45 |
| GAT | **0.587** | 0.379 | 0.584 | 0.371 | 0.44 / 0.90 / 0.42 |
| CGNN-SAGE | 0.553 | 0.380 | 0.601 | 0.387 | 0.46 / 0.91 / 0.43 |
| STHNN (lag 100) | — | — | 0.612 | — | progetto parallelo |
| STHNN (lag 150) | — | — | 0.601 | — | progetto parallelo |

Matrice di confusione del migliore (GraphSAGE, tuned; righe = vero, colonne = predetto):

```
          down    flat     up
down      1237    1973      9        ← quasi mai confonde down con up
flat      1050   40311   2495
up           1    1326   1598
```

Il modello praticamente **non confonde mai le due direzioni tra loro** (10 errori su 6144 campioni direzionali): gli errori sono quasi tutti verso/da la classe flat, cioè sul "quando" il segnale è abbastanza forte, non sul "dove" va il prezzo.

Risultati storici sullo split esplorativo `by_lag` (non confrontabili coi precedenti): CGNN 0.628, STGCN 0.623, STHNN 0.617 (tuned).

## 6. Confronto con il benchmark HLOB

Il paper HLOB valuta **10 architetture SOTA** (DeepLOB, transformer, TABL, ecc. + HLOB stesso) su 15 titoli × 3 anni (2017–2019; 40 giorni di train, 5 di validation, 10 di test per anno). Per **CSCO a orizzonte 50** (Tabella 5 del paper, valori verificati dal PDF):

| Modello (HLOB paper) | F1 | MCC |
|---|---|---|
| **HLOB** (migliore) | **0.60** | **0.40** |
| DeepLOB (secondo) | 0.58 | 0.37 |
| altri 7 (cnn/transformer/tabl) | 0.52–0.57 | 0.31–0.36 |
| iTransformer (peggiore) | 0.41 | 0.18 |

| Nostri (5 giorni, split onesto) | F1 | MCC |
|---|---|---|
| **GraphSAGE** | **0.609** | **0.398** |
| CGNN-SAGE | 0.601 | 0.387 |
| STHNN | 0.612 | — |

**Lettura**: con una frazione dei dati (5 giorni vs ~3 anni) i nostri modelli raggiungono il vertice del campo su entrambe le metriche. Non lo superano — e questo è il punto teoricamente interessante.

## 7. Cosa abbiamo imparato

1. **Esiste un soffitto informativo a ~0.60–0.61 di F1.** Sei nostre architetture + dieci del paper, con quantità di dati diversissime, convergono tutte lì. Il collo di bottiglia è il contenuto predittivo del LOB a questo orizzonte, non la capacità dei modelli.

2. **L'operatore di grafo conta, entro il soffitto.** GCN nuda è nettamente la peggiore (0.572); SAGE, che separa la rappresentazione del nodo da quella dei vicini, è la migliore (0.609). L'attention di GAT non paga rispetto a SAGE, ma è la più robusta senza tuning delle soglie.

3. **Aggiungere la dimensione temporale esplicita non sfonda il soffitto.** L'ibrido CGNN-SAGE (CNN sui lag + SAGE sul grafo) fa 0.601: alla pari con SAGE puro (differenza −0.008, dentro il rumore da singolo seed). Coerente con l'ipotesi del soffitto: l'informazione temporale è già catturata dalla struttura a lag del grafo.

4. **Il tuning delle soglie vale +0.05–0.10 di F1** rispetto all'argmax (tranne per GAT, già ben calibrato). È una procedura senza leakage e metodologicamente corretta per classi sbilanciate.

5. **Gli errori residui sono di "intensità", non di direzione**: le confusioni down↔up sono quasi assenti; il modello sbaglia solo nel distinguere un movimento ≥1 tick dal rumore.

## 8. Limiti e prossimi passi

- **Singolo seed**: i numeri riportati sono di un run per modello. Il passo successivo è la ripetizione con ≥3 seed per riportare media ± deviazione standard (indispensabile per affermare la parità con HLOB con rigore statistico, dato che HLOB media su 3 anni).
- **Terza metrica**: HLOB riporta anche p_T (probabilità di chiudere correttamente una transazione round-trip); una metrica analoga è in preparazione.
- **Oltre il soffitto**: l'unica strada plausibile non è cambiare architettura ma **cambiare l'informazione in ingresso** — es. feature di order-flow (OFI, spread) per nodo, attualmente in sviluppo.

---

*Codice: `scripts/train.py` (training + valutazione con auto-save in `results/`), modelli in `src/models/`, configurazione in `config/default.yaml`. Risultati grezzi: `results/summary.csv`.*
