"""
LOB → GNN (GAT) — Previsione direzione mid-price: down(0) / flat(1) / up(2)
──────────────────────────────────────────────────────────────────────────────
Struttura attesa dei file:
  ADJ_PATH      : CSV  [3000 × 3000]  — adj matrix con pesi volumetrici
  FEATURES_PATH : NPY  [T × 3000 × 2] — (price, volume) per ogni nodo e timestamp
                  oppure CSV [T × 6000] — colonne: price_0, vol_0, price_1, vol_1, ...

Ordinamento nodi assunto:
  bid level i, lag j  →  nodo  i * N_LAGS + j          (i=0..9,  j=0..149)
  ask level i, lag j  →  nodo  N_LAGS*N_LEVELS + i*N_LAGS + j
  → best bid (lag 0): nodo 0     | best ask (lag 0): nodo 1500

Dipendenze:
  pip install torch torch-geometric scikit-learn pandas numpy tqdm
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# CONFIG  ← modifica qui
# ──────────────────────────────────────────────────────────────
ADJ_PATH      = "adj_matrix.csv"
FEATURES_PATH = "node_features.csv"   # oppure "node_features.csv"

N_LEVELS = 10
N_LAGS   = 151
N_SIDES  = 2                                  # bid + ask
N_NODES  = N_LEVELS * N_LAGS * N_SIDES        # 3000

# Soglia label: ±0.01% → flat, altrimenti up/down
THRESHOLD = 0.0001

# Architettura
HIDDEN_DIM  = 64
NUM_HEADS   = 4
NUM_LAYERS  = 3
DROPOUT     = 0.3

# Training
BATCH_SIZE  = 32
LR          = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS      = 100
PATIENCE    = 15                              # early stopping

# Split temporale (no shuffle per time-series!)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = rimanente 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────
# 1. CARICAMENTO ADJ MATRIX → edge_index, edge_weight
# ──────────────────────────────────────────────────────────────
def load_adj(path: str):
    """
    Legge l'adj matrix dal CSV e la converte nel formato COO sparse
    richiesto da PyTorch Geometric.

    Returns:
        edge_index  : [2, E]  LongTensor
        edge_weight : [E]     FloatTensor
    """
    print(f"  Carico adj matrix da {path} ...")
    adj = pd.read_csv(path, header=None).values.astype(np.float32)
    assert adj.shape == (N_NODES, N_NODES), (
        f"Dimensione attesa ({N_NODES},{N_NODES}), trovata {adj.shape}"
    )
    adj_t = torch.tensor(adj)
    edge_index  = adj_t.nonzero(as_tuple=False).t().contiguous()   # [2, E]
    edge_weight = adj_t[edge_index[0], edge_index[1]]               # [E]
    print(f"  Nodi: {N_NODES} | Archi: {edge_index.shape[1]}")
    return edge_index, edge_weight


# ──────────────────────────────────────────────────────────────
# 2. CARICAMENTO FEATURE DEI NODI
# ──────────────────────────────────────────────────────────────
def load_features(path: str) -> np.ndarray:
    """
    Returns:
        features : np.ndarray  [T, N_NODES, 2]  (price, volume)
    """
    print(f"  Carico features da {path} ...")
    if path.endswith(".npy"):
        feat = np.load(path)
        if feat.ndim == 2:
            # Formato [T, N_NODES*2] → reshape
            feat = feat.reshape(-1, N_NODES, 2)
    else:
        raw  = pd.read_csv(path, header=None).values.astype(np.float32)
        feat = raw.reshape(-1, N_NODES, 2)

    assert feat.shape[1] == N_NODES and feat.shape[2] == 2, (
        f"Shape attesa [T,{N_NODES},2], trovata {feat.shape}"
    )
    print(f"  Timestamps: {feat.shape[0]}")
    return feat.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 3. CALCOLO LABEL (mid-price direction t→t+1)
# ──────────────────────────────────────────────────────────────
def compute_labels(features: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """
    best_bid price : nodo 0    (bid level 0, lag 0 = più recente)
    best_ask price : nodo 1500 (ask level 0, lag 0 = più recente)
    mid_price = (best_bid + best_ask) / 2

    Label[t] = direzione di mid_price[t+1] rispetto a mid_price[t]
        0 → down   (Δ < -threshold)
        1 → flat   (|Δ| ≤ threshold)
        2 → up     (Δ >  threshold)
    """
    # Nuovi indici basati sul file CSV:
    ASK_L0_LAG0 = 0                           # Primo nodo: ask_0_lag_0
    BID_L0_LAG0 = N_LEVELS * N_LAGS           # Nodo 1510: bid_0_lag_0

    best_ask  = features[:, ASK_L0_LAG0, 0] # price al channel 0
    best_bid  = features[:, BID_L0_LAG0, 0]
    mid_price = (best_bid + best_ask) / 2.0

    pct_change = (mid_price[1:] - mid_price[:-1]) / (mid_price[:-1] + 1e-10)

    labels = np.ones(len(pct_change), dtype=np.int64)   # default: flat
    labels[pct_change >  threshold] = 2                  # up
    labels[pct_change < -threshold] = 0                  # down

    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: "down", 1: "flat", 2: "up"}
    dist = {label_names[k]: int(v) for k, v in zip(unique, counts)}
    print(f"  Distribuzione label: {dist}")
    return labels


# ──────────────────────────────────────────────────────────────
# 4. COSTRUZIONE DATASET (lista di Data PyG)
# ──────────────────────────────────────────────────────────────
def build_dataset(edge_index, edge_weight, features, labels):
    """
    Crea un oggetto Data per ogni timestamp t = 0..T-2.
    Le feature vengono normalizzate globalmente (fit sul solo train set
    non è possibile senza conoscere lo split, quindi si fa un fit globale
    — in produzione fare fit solo sul train).
    """
    T    = len(labels)
    feat = features[:T]      # [T, N, 2]

    # Normalizzazione separata per prezzo e volume
    prices  = feat[:, :, 0].reshape(-1, 1)
    volumes = feat[:, :, 1].reshape(-1, 1)

    sc_p = StandardScaler().fit(prices)
    sc_v = StandardScaler().fit(volumes)

    prices_n  = sc_p.transform(prices).reshape(T, N_NODES)
    volumes_n = sc_v.transform(volumes).reshape(T, N_NODES)

    dataset = []
    for t in range(T):
        x = torch.tensor(
            np.stack([prices_n[t], volumes_n[t]], axis=1),
            dtype=torch.float32
        )  # [N_NODES, 2]
        y = torch.tensor(labels[t], dtype=torch.long)
        dataset.append(Data(x=x, edge_index=edge_index,
                            edge_attr=edge_weight, y=y))

    return dataset, sc_p, sc_v


# ──────────────────────────────────────────────────────────────
# 5. MODELLO — GAT multi-layer
# ──────────────────────────────────────────────────────────────
class LOB_GAT(nn.Module):
    """
    Stack di GATConv layers + global mean pooling + MLP classifier.

    Scelta di GAT per questo grafo:
    - Localmente denso → l'attention impara a pesare i vicini rilevanti
    - Asimmetria bid/ask naturalmente gestita da pesi diversi per direzione
    - edge_attr (volumi) usati come feature degli archi
    """

    def __init__(
        self,
        in_dim:      int = 2,
        hidden_dim:  int = HIDDEN_DIM,
        num_heads:   int = NUM_HEADS,
        num_layers:  int = NUM_LAYERS,
        num_classes: int = 3,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.dropout_p = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Layer 0: in_dim → hidden_dim * num_heads
        self.convs.append(
            GATConv(in_dim, hidden_dim, heads=num_heads,
                    dropout=dropout, edge_dim=1, concat=True)
        )
        self.norms.append(nn.LayerNorm(hidden_dim * num_heads))

        # Layer 1..N-2 (se num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                        dropout=dropout, edge_dim=1, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))

        # Layer finale: average heads → [N, hidden_dim]
        self.convs.append(
            GATConv(hidden_dim * num_heads, hidden_dim, heads=1,
                    dropout=dropout, edge_dim=1, concat=False)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # edge_attr deve essere [E, 1]
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Readout: media su tutti i nodi del grafo
        x = global_mean_pool(x, batch)      # [batch_size, hidden_dim]
        return self.head(x)                 # [batch_size, num_classes]


# ──────────────────────────────────────────────────────────────
# 6. TRAINING / EVALUATION
# ──────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(DEVICE)
            out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            correct    += (out.argmax(1) == batch.y).sum().item()
            total      += batch.num_graphs

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_full(model, loader):
    """Report dettagliato per il test set."""
    model.eval()
    all_preds, all_true = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).argmax(1)
        all_preds.extend(pred.cpu().numpy())
        all_true.extend(batch.y.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_true, all_preds,
                                 target_names=["down", "flat", "up"],
                                 digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_true, all_preds))


# ──────────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────────
def main():
    print(f"\n{'─'*60}")
    print(f"  LOB-GNN | Dispositivo: {DEVICE}")
    print(f"{'─'*60}\n")

    # ── Caricamento ──
    print("[1/4] Caricamento dati...")
    edge_index, edge_weight = load_adj(ADJ_PATH)
    features = load_features(FEATURES_PATH)
    labels   = compute_labels(features, THRESHOLD)

    # ── Dataset ──
    print("\n[2/4] Costruzione dataset...")
    dataset, sc_p, sc_v = build_dataset(edge_index, edge_weight, features, labels)

    T  = len(dataset)
    t1 = int(T * TRAIN_RATIO)
    t2 = int(T * (TRAIN_RATIO + VAL_RATIO))

    train_ds, val_ds, test_ds = dataset[:t1], dataset[t1:t2], dataset[t2:]
    print(f"  Split → train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # ── Class weights (bilancio label) ──
    train_labels = np.array([d.y.item() for d in train_ds])
    counts       = np.bincount(train_labels, minlength=3).astype(np.float32)
    weights      = torch.tensor(1.0 / (counts + 1e-6))
    weights      = (weights / weights.sum() * 3).to(DEVICE)   # normalizzato a media=1

    # ── Modello ──
    print("\n[3/4] Inizializzazione modello...")
    model     = LOB_GAT().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=PATIENCE // 2, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=weights)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametri trainabili: {n_params:,}")

    # ── Training ──
    print("\n[4/4] Training...\n")
    best_val_loss = float("inf")
    patience_cnt  = 0

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   optimizer, criterion, train=False)
        scheduler.step(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_cnt  = 0
            torch.save(model.state_dict(), "best_lob_gnn.pt")
        else:
            patience_cnt += 1

        if epoch % 10 == 0:
            tqdm.write(
                f"Ep {epoch:03d} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.3f}"
            )

        if patience_cnt >= PATIENCE:
            tqdm.write(f"\nEarly stopping a epoch {epoch}.")
            break

    # ── Valutazione finale ──
    model.load_state_dict(torch.load("best_lob_gnn.pt", map_location=DEVICE))
    _, test_acc = run_epoch(model, test_loader, None, criterion, train=False)
    print(f"\nTest accuracy: {test_acc:.4f}")
    evaluate_full(model, test_loader)


if __name__ == "__main__":
    main()