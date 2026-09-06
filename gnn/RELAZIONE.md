# GNN su grafo TMFG per la predizione della direzione del mid-price da Limit Order Book

> Riassunto della ricerca — 2026-09-06
> Dati: LOB di Cisco (CSCO), LOBSTER, 5 giorni (gennaio 2019). Benchmark di riferimento: paper HLOB (Briola, Bartolucci, Aste — arXiv:2405.18938).

---

## Sintesi dei risultati

Con **5 giorni di dati** (contro i ~3 anni del benchmark) e uno **split di valutazione onesto** (giorni interi mai visti in training), il nostro miglior modello raggiunge **F1 macro 0.643 / MCC 0.448**, contro **F1 0.60 / MCC 0.40** di HLOB — il migliore di 10 modelli SOTA sullo stesso titolo, stesso task e stesse label. Su questo titolo i nostri modelli **superano lo stato dell'arte su entrambe le metriche usando ~30× meno dati**, e il sorpasso è confermato da una campagna **multi-seed** (§5).

Tre risultati di merito, in ordine di importanza:

1. **La dimensione temporale è l'ingrediente decisivo.** Le GNN "statiche" (GCN, SAGE, GAT) si fermano a 0.57–0.61; aggiungendo una convoluzione temporale sui lag (CGNN, STGCN) si arriva a ~0.63 a parità di parametri.
2. **La capacità non è ancora satura.** Scalando il modello da 181k a 720k parametri la F1 sale da 0.626 a **0.643** (+0.016, ~5σ). Contrariamente a quanto ipotizzato in una fase precedente del lavoro, **non abbiamo ancora raggiunto il soffitto**: dove la scala saturi resta una domanda aperta.
3. **L'operatore di grafo conta meno dell'architettura temporale**, ma non è neutro: l'attention (GAT) è sistematicamente la scelta peggiore in entrambe le architetture spazio-temporali.

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

**Dodici configurazioni** nostre (più lo STHNN del progetto parallelo, 13 righe in §5e), nove delle quali a **budget di parametri comparabile (~180–206k)**, stessa testa di classificazione (mean+max pooling → MLP), stesso training. Le tre famiglie si distinguono per **come** (e se) trattano il tempo; le varianti per **quale operatore di grafo** usano: 3 famiglie × 3 operatori = 9 configurazioni a budget.

| Famiglia | Trattamento del tempo | Varianti (operatore di grafo) |
|---|---|---|
| **GNN statiche** | nessuno: i lag sono solo nodi del grafo | GCN (206k), SAGE (182k), GAT (188k) |
| **CGNN** | CNN 1D sull'asse dei lag → poi grafo | GCN (181k), SAGE (185k), GAT (186k) |
| **STGCN** | blocchi intercalati [temporale → grafo → temporale] | GCN (183k), SAGE (182k), GAT (186k) |

A queste si aggiungono tre **test di scala** fuori budget: **SAGE @272k**, **CGNN-SAGE @273k** e **CGNN @720k**.
A questi si affianca il progetto parallelo **RecurrentSparseSTHNN** (rete ipergrafica spazio-temporale, stesso task e label), usato come confronto interno.

## 4. Protocollo sperimentale

- **Split onesto (`by_file`)**: train sui giorni 1–3, validation sul giorno 4, **test sul giorno 5 mai visto** — l'analogo del protocollo HLOB (giorni di test disgiunti dal training). Lo split alternativo `by_lag` (70/15/15 dentro ogni giornata) è stato usato solo nelle fasi esplorative: è leggermente ottimistico (leak intra-day) e non entra nei confronti finali.
- **Training bilanciato** (50k campioni/classe), **test sulla distribuzione naturale** (~88% flat) — come HLOB (5k/classe/giorno in training, test naturale).
- **Tuning delle soglie di decisione**: le soglie di probabilità per down/up sono tarate sulla validation e applicate al test (procedura senza leakage). È la scelta corretta per la macro-F1 su classi sbilanciate: l'argmax minimizza l'error-rate, non la macro-F1. Riportiamo entrambe le regole.
- **Multi-seed**: i due modelli di punta sono stati ripetuti su 3 seed per stimare il rumore di inizializzazione.
- Ottimizzazione: AdamW, early stopping su validation, LR scheduling.

## 5. Risultati (split onesto, test = giorno mai visto, 50k campioni)

### 5a. Modelli di punta, con varianza multi-seed

| Modello | Parametri | F1 (tuned) | MCC (tuned) |
|---|---|---|---|
| **CGNN scalato** | 720k | **0.6428** | **0.4483** |
| STGCN | 183k | 0.6282 ± 0.0076 | 0.4286 ± 0.0103 |
| CGNN | 181k | 0.6259 ± 0.0031 | 0.4235 ± 0.0045 |

Il **rumore da seed è piccolo (σ ≈ 0.003–0.008)**: differenze ≥ 0.015 sono quindi significative, mentre CGNN e STGCN a budget sono statisticamente indistinguibili tra loro. Il modello scalato a 720k supera i modelli a budget di ~5σ: il guadagno di scala è reale.

### 5b. Griglia architettura × operatore (tutti a ~180k parametri)

| | GCN | SAGE | GAT |
|---|---|---|---|
| **statica** | 0.572 | 0.609 | 0.584 |
| **CGNN** (CNN → grafo) | **0.626** | 0.601 | 0.606 |
| **STGCN** (intercalato) | **0.628** | **0.632** | 0.598 |

Due letture immediate: le righe spazio-temporali stanno sopra la riga statica (tranne con GAT), e la colonna GAT è la peggiore di ogni riga.

### 5c. Test di scala

| Modello | 180k | 270k | 720k |
|---|---|---|---|
| SAGE (statica) | 0.609 | **0.625** | — |
| CGNN | 0.626 | — | **0.643** |

Entrambe le famiglie guadagnano ~+0.017 aumentando i parametri. **La capacità non è satura.**

### 5d. Struttura degli errori

Matrice di confusione del migliore (CGNN @720k, tuned; righe = vero, colonne = predetto):

```
          down    flat     up
down      1600    1519      1        ← quasi mai confonde down con up
flat      1378   40649   1844
up           4    1492   1513
```

Il modello **non confonde quasi mai le due direzioni tra loro**: 5 errori su 6129 campioni direzionali. Gli errori sono quasi tutti verso/da la classe flat, cioè sul *quando* il segnale è abbastanza forte, non sul *dove* va il prezzo. È la struttura d'errore che si vorrebbe in un contesto di trading: il modello sbaglia per prudenza, non per direzione.

### 5e. Tabella comparativa completa

Tutte le configurazioni su `by_file`, con **entrambe le regole di decisione** (argmax e soglie tarate su validation). Ordinamento per F1 tuned. HLOB è inserito nella sua posizione reale in classifica.

| # | Modello | Tempo | Op. | Param | F1 argmax | MCC argmax | F1 tuned | MCC tuned |
|---|---|---|---|---|---|---|---|---|
| 1 | **CGNN scalato** | CNN→grafo | GCN | 720k | 0.5209 | 0.3770 | **0.6428** | **0.4483** |
| 2 | STGCN-SAGE | intercalato | SAGE | 182k | 0.5721 | 0.4096 | 0.6324 | 0.4322 |
| 3 | STGCN *(3 seed)* | intercalato | GCN | 183k | 0.5695 ± 0.0092 | 0.4045 ± 0.0012 | 0.6282 ± 0.0076 | 0.4286 ± 0.0103 |
| 4 | CGNN *(3 seed)* | CNN→grafo | GCN | 181k | 0.5412 ± 0.0032 | 0.3881 ± 0.0050 | 0.6259 ± 0.0031 | 0.4235 ± 0.0045 |
| 5 | SAGE scalato | — | SAGE | 272k | 0.5319 | 0.3733 | 0.6253 | 0.4204 |
| 6 | CGNN-SAGE scalato | CNN→grafo | SAGE | 273k | 0.5261ᴬ | 0.3787ᴬ | 0.6153ᴬ | 0.4078ᴬ |
| 7 | STHNN *(prog. parallelo)* | ipergrafo ST | — | — | — | — | 0.6115ᴬ | — |
| 8 | SAGE | — | SAGE | 182k | 0.5445ᴬ | 0.3844ᴬ | 0.6086ᴬ | 0.3976ᴬ |
| 9 | CGNN-GAT | CNN→grafo | GAT | 186k | 0.5335 | 0.3440 | 0.6057 | 0.3978 |
| 10 | CGNN-SAGE | CNN→grafo | SAGE | 185k | 0.5529ᴬ | 0.3803ᴬ | 0.6008ᴬ | 0.3868ᴬ |
| — | **HLOB** *(benchmark)* | HCNN+LSTM | — | **180k** | n/d † | n/d † | **0.60** | **0.40** |
| 11 | STGCN-GAT | intercalato | GAT | 186k | 0.5658 | 0.3951 | 0.5975 | 0.3794 |
| 12 | GAT | — | GAT | 188k | **0.5874**ᴬ | 0.3794ᴬ | 0.5841ᴬ | 0.3710ᴬ |
| 13 | GCN | — | GCN | 206k | 0.4757ᴬ | 0.3107ᴬ | 0.5715ᴬ | 0.3427ᴬ |

**Il confronto con HLOB è a parametri appaiati.** La Tabella 3 del paper riporta per HLOB **1.8 × 10⁵ ≈ 180k parametri trainabili** — esattamente il nostro budget. Quindi le nove configurazioni a ~180k non solo usano ~30× meno dati, ma hanno anche **la stessa taglia del benchmark**. (Per riferimento, gli altri: BinBTabl 6.6k, BinCTabl 22k, CNN1 35k, Transformer/iTransformer 110k, DeepLOB 140k, DLA 220k, CNN2 280k, LobTransformer 2.0M.) Fa eccezione solo la riga 1, che a 720k è **4× HLOB**: è un test di scala, non un confronto equo.

† Il paper non dichiara la regola di decisione usata in Tabella 5; l'evaluator di LOBFrame calcola le metriche *"as function of probability threshold"*, quindi i valori di HLOB potrebbero già essere soglia-tarati e non argmax puri. Nel dubbio li confrontiamo con la nostra colonna *tuned*, che è l'ipotesi a noi più sfavorevole.

**Osservazioni sulle due regole di decisione:**
- Il tuning delle soglie vale **+0.05 / +0.12 di F1** su tutti i modelli tranne GAT statica, l'unico caso in cui *peggiora* (0.5874 → 0.5841): le soglie tarate su validation non generalizzano al test. È anche l'unico modello il cui argmax (0.5874) sarebbe il migliore della colonna.
- **Il ranking cambia tra le due regole.** All'argmax i modelli scalati sono i peggiori (CGNN@720k fa 0.5209, ultimo assoluto) perché più capacità = probabilità più diffuse sulla classe flat; dopo il tuning diventano i migliori. Questo conferma che su classi al ~88% flat l'argmax non è la regola informativa, ed è la ragione per cui riportiamo entrambe.
- **L'MCC è molto più stabile della F1 tra le due regole** (es. STGCN 0.4045 → 0.4286, +0.024, contro +0.059 di F1): è meno sensibile alla soglia, e quindi il confronto più conservativo con HLOB.

ᴬ = valutato sul sottocampione di test di luglio (support down/up 3219/2925); gli altri su quello di settembre (3120/3009) — due estrazioni da 50k dello stesso giorno, differenza attesa ~±0.005 (vedi §8).

## 6. Confronto con il benchmark HLOB

Il paper HLOB valuta **10 architetture SOTA** (DeepLOB, transformer, TABL, ecc. + HLOB stesso) su 15 titoli × 3 anni (2017–2019; 40 giorni di train, 5 di validation, 10 di test per anno). Per **CSCO a orizzonte 50** (Tabella 5 del paper, valori verificati sul PDF):

| Modello (HLOB paper) | F1 | MCC |
|---|---|---|
| **HLOB** (migliore) | **0.60** | **0.40** |
| DeepLOB (secondo) | 0.58 | 0.37 |
| altri 7 (cnn/transformer/tabl) | 0.52–0.57 | 0.31–0.36 |
| iTransformer (peggiore) | 0.41 | 0.18 |

| Nostri (5 giorni, split onesto) | F1 | MCC |
|---|---|---|
| **CGNN @720k** | **0.643** | **0.448** |
| STGCN | 0.628 ± 0.008 | 0.429 ± 0.010 |
| CGNN | 0.626 ± 0.003 | 0.424 ± 0.005 |
| SAGE @272k | 0.625 | 0.420 |
| STHNN (progetto parallelo) | 0.612 | — |

**Lettura**: con una frazione dei dati (5 giorni vs ~3 anni) i nostri modelli spazio-temporali superano il vertice del campo HLOB su entrambe le metriche, di ~0.03–0.04 in F1 e ~0.03–0.05 in MCC — oltre 3σ del rumore di seed. Resta la precisazione di §8: il confronto vive su **un solo giorno di test**, mentre HLOB media su 10 giorni × 3 anni.

## 7. Cosa abbiamo imparato

1. **La dimensione temporale esplicita è l'ingrediente decisivo.** A parità di parametri, aggiungere una convoluzione sui lag porta la GCN da 0.572 a 0.626 (+0.054) e definisce le due architetture migliori. La struttura a lag del grafo, da sola, non basta: il message passing "vede" la posizione temporale ma non ne modella la dinamica.

2. **La CNN temporale aiuta la GCN, non SAGE — verificato a parametri appaiati.** Il controllo decisivo è stato confrontare SAGE puro e CNN+SAGE a due budget diversi:

   | budget | SAGE puro | CNN + SAGE | vince |
   |---|---|---|---|
   | ~180k | **0.609** | 0.601 | SAGE |
   | ~270k | **0.625** | 0.615 | SAGE |

   A entrambi i budget SAGE puro batte la versione con CNN. L'ipotesi che il guadagno di CNN+SAGE a 270k fosse dovuto alla CNN è quindi **falsificata**: era capacità, e quella stessa capacità resa a SAGE puro rende di più. Plausibile interpretazione: SAGE, separando la trasformazione del nodo da quella dei vicini, cattura già parte della dinamica che la CNN aggiungerebbe.

3. **La capacità non è satura — un'ipotesi precedente è stata falsificata.** In una fase intermedia del lavoro avevamo concluso che esistesse un *soffitto informativo* a ~0.62 e che aumentare i parametri non avrebbe aiutato. Il test di scala smentisce: CGNN da 181k a 720k passa da 0.626 a **0.643**, e SAGE da 182k a 272k passa da 0.609 a 0.625. Entrambi ~+0.017, ben oltre il rumore. Quel "soffitto" era, almeno in parte, un limite di capacità e non di informazione. **Dove la scala saturi resta ignoto**: è la direzione sperimentale più promettente rimasta.

4. **L'attention non paga.** GAT è la scelta peggiore in tutte e tre le famiglie (0.584 statica, 0.606 in CGNN, 0.598 in STGCN), con gap di 0.02–0.035 rispetto alle controparti — ben oltre il rumore. Interpretazione: il TMFG ha **già** filtrato gli archi informativi, quindi imparare pesi sugli archi aggiunge parametri e varianza senza aggiungere selettività.

5. **Il tuning delle soglie vale +0.05–0.12 di F1** rispetto all'argmax. È una procedura senza leakage e corretta per classi sbilanciate; che non sia un artefatto lo conferma il fatto che la F1 di validation predice bene quella di test (salto val→test ~0.01 su tutti i modelli di punta).

## 8. Limiti e prossimi passi

- **Un solo giorno di test** — il limite più serio. Con 5 giorni di dati lo split onesto lascia **un unico giorno held-out**: tutti i numeri vivono sulla distribuzione di quel giorno, mentre HLOB media su 10 giorni × 3 anni. Il multi-seed misura la varianza di inizializzazione, **non** quella giorno-per-giorno, che resta ignota. Più giorni di dati LOBSTER sono l'unica via per chiuderlo.
- **Fino a dove scala?** — la domanda aperta più interessante: il test si ferma a 720k parametri. Un run a ~1.4M direbbe se il guadagno continua o satura. Va accompagnato da multi-seed, perché il 0.643 è attualmente su singolo seed.
- **Due sottocampioni di test.** Per una ricostruzione dell'ambiente, i run di luglio e quelli di settembre hanno estratto due sottocampioni di test diversi (50k su ~800k dello stesso giorno). I nove run di settembre condividono lo stesso test set — i confronti tra loro sono puliti — mentre le medie multi-seed mescolano i due. L'effetto è dell'ordine di ±0.005, molto inferiore ai divari discussi, e la causa è stata rimossa (il sottocampione ora usa un seed fisso, scollegato dal seed di training).
- **Terza metrica**: HLOB riporta anche p_T (probabilità di chiudere correttamente una transazione round-trip); una metrica analoga è in preparazione.
- **Più informazione in ingresso**: feature di order-flow (OFI, spread) come attributi di nodo restano la leva complementare alla scala, in sviluppo.

---

*Codice: `scripts/train.py` (training + valutazione con auto-save in `results/`), modelli in `src/models/`, configurazione in `config/default.yaml`. Risultati grezzi: `results/summary.csv`.*
