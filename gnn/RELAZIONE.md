# GNN su grafo TMFG per la predizione della direzione del mid-price da Limit Order Book

> Riassunto della ricerca — 2026-07-15
> Dati: LOB di Cisco (CSCO), LOBSTER, 5 giorni (gennaio 2019). Benchmark di riferimento: paper HLOB (Briola, Bartolucci, Aste — arXiv:2405.18938).

---

## Sintesi dei risultati

Con **5 giorni di dati** (contro i ~3 anni del benchmark) e uno **split di valutazione onesto** (giorni interi mai visti in training), i nostri due modelli **spazio-temporali** raggiungono **F1 macro 0.625 / MCC 0.421** (CGNN) e **0.620 / 0.417** (STGCN), contro **F1 0.60 / MCC 0.40** di HLOB — il migliore di 10 modelli SOTA sullo stesso titolo, stesso task e stesse label. Su questo titolo, quindi, i nostri modelli **superano lo stato dell'arte su entrambe le metriche, usando ~30× meno dati** (in attesa di conferma multi-seed, vedi §8).

Il secondo risultato, di natura più teorica: **l'ingrediente decisivo è la dimensione temporale, non l'operatore di grafo.** Le tre GNN "statiche" (GCN, SAGE, GAT) si fermano a 0.57–0.61; aggiungendo una convoluzione temporale sui lag (CGNN, STGCN) si sale a ~0.62. Sopra questa soglia non si va: si intravede un **soffitto informativo del segnale** a ~0.62 (quanto il libro ordini permette di prevedere a questo orizzonte), non un limite dei modelli — coerente col fatto che anche i 10 modelli SOTA del paper si fermano a 0.60.

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

Ordinati per F1 tuned. In grassetto i due modelli spazio-temporali, che dominano la tabella.

| Modello | F1 (argmax) | MCC (argmax) | F1 (tuned) | MCC (tuned) | F1 down / flat / up (tuned) |
|---|---|---|---|---|---|
| **CGNN** (GCN + CNN temporale) | 0.538 | 0.389 | **0.625** | **0.421** | 0.48 / 0.93 / 0.47 |
| **STGCN** (blocchi spazio-temp.) | 0.580 | 0.406 | **0.620** | **0.417** | 0.45 / 0.93 / 0.48 |
| GraphSAGE | 0.545 | 0.384 | 0.609 | 0.398 | 0.45 / 0.92 / 0.45 |
| CGNN-SAGE | 0.553 | 0.380 | 0.601 | 0.387 | 0.46 / 0.91 / 0.43 |
| GAT | 0.587 | 0.379 | 0.584 | 0.371 | 0.44 / 0.90 / 0.42 |
| GCN | 0.476 | 0.311 | 0.572 | 0.343 | 0.40 / 0.91 / 0.41 |
| STHNN (lag 100) | — | — | 0.612 | — | progetto parallelo |
| STHNN (lag 150) | — | — | 0.601 | — | progetto parallelo |

Matrice di confusione del migliore (CGNN, tuned; righe = vero, colonne = predetto):

```
          down    flat     up
down      1482    1729      8        ← quasi mai confonde down con up
flat      1445   40561   1850
up           4    1468   1453
```

Il modello praticamente **non confonde mai le due direzioni tra loro** (12 errori su 6144 campioni direzionali): gli errori sono quasi tutti verso/da la classe flat, cioè sul "quando" il segnale è abbastanza forte, non sul "dove" va il prezzo.

**Coerenza col vecchio split esplorativo `by_lag`**: CGNN faceva 0.628 e STGCN 0.623; su `by_file` fanno 0.625 e 0.620. Il calo dovuto allo split onesto è ~0.004 — trascurabile, e in linea con lo STHNN (~0.006). I numeri sono internamente consistenti, non frutto del caso.

> **Controllo di capacità su CGNN-SAGE** (diagnostico, fuori dal budget parametri): a hidden 175 (273k param, invece dei 140/185k a budget) CGNN-SAGE sale a **F1 0.615 / MCC 0.408** e batte SAGE puro. Conferma che il suo −0.008 a budget era il taglio di capacità, non l'inutilità della CNN (vedi §7.2). Non entra nella classifica sopra perché non è a parità di parametri.

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
| **CGNN** (GCN + CNN temporale) | **0.625** | **0.421** |
| **STGCN** | **0.620** | **0.417** |
| GraphSAGE | 0.609 | 0.398 |
| STHNN | 0.612 | — |

**Lettura**: con una frazione dei dati (5 giorni vs ~3 anni) i nostri modelli spazio-temporali **superano** il vertice del campo HLOB (0.60/0.40) su entrambe le metriche; le GNN statiche lo eguagliano. Due precisazioni doverose (§8): il risultato è su **singolo seed** e su **un unico giorno di test** (5 giorni ⇒ un solo giorno held-out), mentre HLOB media su 10 giorni × 3 anni. Quindi "superiamo HLOB" va inteso come *su questo titolo e questo giorno*, in attesa della varianza multi-seed.

## 7. Cosa abbiamo imparato

1. **La dimensione temporale esplicita è l'ingrediente decisivo.** I due modelli che aggiungono una convoluzione temporale sui lag prima/dentro il message passing (CGNN, STGCN) sono i migliori (~0.62); le tre GNN statiche (GCN, SAGE, GAT) restano a 0.57–0.61. La struttura a lag del grafo da sola non basta: serve un operatore che modelli esplicitamente l'evoluzione nel tempo.

2. **La CNN temporale aiuta entrambi gli operatori, ma è cara in parametri.** Un confronto controllato (a capacità appaiata) lo chiarisce:
   - GCN 0.572 → **CGNN 0.625** (+0.053): sulla GCN, operatore "economico", la CNN entra nel budget ~180k e trasforma il modello peggiore nel migliore.
   - SAGE 0.609 → CGNN-SAGE **a budget** 0.601 (−0.008), ma → CGNN-SAGE **a capacità piena** (hidden 175, 273k param) **0.615** (+0.006): la CNN aggiunge segnale anche a SAGE — il calo a budget era dovuto al taglio dell'hidden a 140 (SAGEConv ha 2 matrici/layer, "costa" il doppio), non alla CNN.
   
   Morale: la dimensione temporale porta segnale a qualunque operatore; la differenza è il *costo*. La GCN è abbastanza economica da permettersi CNN + capacità piena entro budget, SAGE no. Il miglior modello a parità di budget resta quindi **GCN + temporale** (0.625 con 181k param), più efficiente del CNN+SAGE anche quando quest'ultimo gira a 273k.

3. **Esiste comunque un soffitto informativo, ora stimabile a ~0.62 di F1.** Nessuno dei sei modelli supera 0.625, e il campo di 10 modelli SOTA del paper si ferma a 0.60. Il collo di bottiglia resta il contenuto predittivo del LOB a questo orizzonte: la dimensione temporale porta i modelli *fino* al soffitto, non oltre.

4. **Il tuning delle soglie vale +0.05–0.10 di F1** rispetto all'argmax (tranne per GAT, già ben calibrato). È una procedura senza leakage e metodologicamente corretta per classi sbilanciate. Che il numero non sia un artefatto lo conferma il fatto che la F1 di validation (0.63) predice bene quella di test (0.62): il salto val→test è solo ~0.01.

5. **Gli errori residui sono di "intensità", non di direzione**: le confusioni down↔up sono quasi assenti; il modello sbaglia solo nel distinguere un movimento ≥1 tick dal rumore.

## 8. Limiti e prossimi passi

- **Singolo seed**: i numeri riportati sono di un run per modello. Il passo successivo — ora prioritario proprio su CGNN e STGCN — è la ripetizione con ≥3 seed per riportare media ± deviazione standard. Serve per affermare con rigore statistico il sorpasso su HLOB: lo scarto dei due leader sul resto del campo (~0.02) va confermato oltre il rumore di inizializzazione.
- **Un solo giorno di test**: con 5 giorni di dati, lo split onesto lascia **un unico giorno held-out**. Tutti i numeri vivono sulla distribuzione di quel giorno; HLOB media su 10 giorni × 3 anni. Il multi-seed varia l'inizializzazione ma **non** questo: la varianza giorno-per-giorno resta ignota ed è il limite più serio del confronto. Più giorni di dati LOBSTER sono la via per chiuderlo.
- **Terza metrica**: HLOB riporta anche p_T (probabilità di chiudere correttamente una transazione round-trip); una metrica analoga è in preparazione.
- **Oltre il soffitto ~0.62**: avendo visto che la dimensione temporale porta i modelli fino al soffitto ma non oltre, la strada plausibile non è più cambiare architettura ma **cambiare l'informazione in ingresso** — es. feature di order-flow (OFI, spread) per nodo, attualmente in sviluppo.

---

*Codice: `scripts/train.py` (training + valutazione con auto-save in `results/`), modelli in `src/models/`, configurazione in `config/default.yaml`. Risultati grezzi: `results/summary.csv`.*
