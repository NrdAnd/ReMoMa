"""Historical GAT prototype, retained for source-level research history.

This prototype uses global normalization and differs from the supported pipeline.
Its configured node count is 3020 (2 sides, 10 levels, 151 lag positions).
Use scripts/train.py for current experiments; see archive/README.md.
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
# CONFIGURATION (historical constants)
# ──────────────────────────────────────────────────────────────
ADJ_PATH      = "adj_matrix.csv"
FEATURES_PATH = "node_features.csv"   # or "node_features.csv"

N_LEVELS = 10
N_LAGS   = 151
N_SIDES  = 2                                  # bid + ask
N_NODES  = N_LEVELS * N_LAGS * N_SIDES        # 3000

# Label threshold: ±0.01%; otherwise up/down
THRESHOLD = 0.0001

# Architecture
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

# Chronological split (preserve temporal order)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = remaining 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────
# 1. LOAD ADJACENCY MATRIX → edge_index, edge_weight
# ──────────────────────────────────────────────────────────────
def load_adj(path: str):
    """
    Read CSV adjacency and convert it to PyG sparse COO format.

    Returns:
        edge_index  : [2, E]  LongTensor
        edge_weight : [E]     FloatTensor
    """
    print(f"  Loading adjacency from {path} ...")
    adj = pd.read_csv(path, header=None).values.astype(np.float32)
    assert adj.shape == (N_NODES, N_NODES), (
        f"Expected shape ({N_NODES},{N_NODES}), found {adj.shape}"
    )
    adj_t = torch.tensor(adj)
    edge_index  = adj_t.nonzero(as_tuple=False).t().contiguous()   # [2, E]
    edge_weight = adj_t[edge_index[0], edge_index[1]]               # [E]
    print(f"  Nodes: {N_NODES} | Edges: {edge_index.shape[1]}")
    return edge_index, edge_weight


# ──────────────────────────────────────────────────────────────
# 2. LOAD NODE FEATURES
# ──────────────────────────────────────────────────────────────
def load_features(path: str) -> np.ndarray:
    """
    Returns:
        features : np.ndarray  [T, N_NODES, 2]  (price, volume)
    """
    print(f"  Loading features from {path} ...")
    if path.endswith(".npy"):
        feat = np.load(path)
        if feat.ndim == 2:
            # Format [T, N_NODES*2] → reshape
            feat = feat.reshape(-1, N_NODES, 2)
    else:
        raw  = pd.read_csv(path, header=None).values.astype(np.float32)
        feat = raw.reshape(-1, N_NODES, 2)

    assert feat.shape[1] == N_NODES and feat.shape[2] == 2, (
        f"Expected shape [T,{N_NODES},2], found {feat.shape}"
    )
    print(f"  Timestamps: {feat.shape[0]}")
    return feat.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 3. COMPUTE LABELS (mid-price direction t→t+1)
# ──────────────────────────────────────────────────────────────
def compute_labels(features: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """
    best_bid price : node N_LEVELS * N_LAGS (bid level 0, lag 0)
    best_ask price : node 0 (ask level 0, lag 0)
    mid_price = (best_bid + best_ask) / 2

    Label[t] = direction of mid_price[t+1] relative to mid_price[t]
        0 → down   (Δ < -threshold)
        1 → flat   (|Δ| ≤ threshold)
        2 → up     (Δ >  threshold)
    """
    # Indices used by the CSV input:
    ASK_L0_LAG0 = 0                           # First node: ask_0_lag_0
    BID_L0_LAG0 = N_LEVELS * N_LAGS           # Node 1510: bid_0_lag_0

    best_ask  = features[:, ASK_L0_LAG0, 0] # price in channel 0
    best_bid  = features[:, BID_L0_LAG0, 0]
    mid_price = (best_bid + best_ask) / 2.0

    pct_change = (mid_price[1:] - mid_price[:-1]) / (mid_price[:-1] + 1e-10)

    labels = np.ones(len(pct_change), dtype=np.int64)   # default: flat
    labels[pct_change >  threshold] = 2                  # up
    labels[pct_change < -threshold] = 0                  # down

    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: "down", 1: "flat", 2: "up"}
    dist = {label_names[k]: int(v) for k, v in zip(unique, counts)}
    print(f"  Label distribution: {dist}")
    return labels


# ──────────────────────────────────────────────────────────────
# 4. BUILD DATASET (list of PyG Data objects)
# ──────────────────────────────────────────────────────────────
def build_dataset(edge_index, edge_weight, features, labels):
    """
    Create a Data object for each timestamp t = 0..T-2.
    Historical behavior: normalization is fitted globally before splitting.
    This leaks evaluation information; the supported pipeline fixes this.
    """
    T    = len(labels)
    feat = features[:T]      # [T, N, 2]

    # Separate price and volume normalization
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
# 5. MODEL — GAT multi-layer
# ──────────────────────────────────────────────────────────────
class LOB_GAT(nn.Module):
    """
    GATConv layers, global mean pooling, and an MLP classifier.

    Attention weights neighbor messages; edge_attr provides volume edge features.
    This is a historical prototype, not a validated current model.
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

        # Layer 1..N-2 (if num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                        dropout=dropout, edge_dim=1, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))

        # Final layer: average heads → [N, hidden_dim]
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
        # edge_attr must be [E, 1]
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Readout: mean over graph nodes
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
    """Print a detailed test-set report."""
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
    print(f"  LOB-GNN | Device: {DEVICE}")
    print(f"{'─'*60}\n")

    # ── Loading ──
    print("[1/4] Loading data...")
    edge_index, edge_weight = load_adj(ADJ_PATH)
    features = load_features(FEATURES_PATH)
    labels   = compute_labels(features, THRESHOLD)

    # ── Dataset ──
    print("\n[2/4] Building dataset...")
    dataset, sc_p, sc_v = build_dataset(edge_index, edge_weight, features, labels)

    T  = len(dataset)
    t1 = int(T * TRAIN_RATIO)
    t2 = int(T * (TRAIN_RATIO + VAL_RATIO))

    train_ds, val_ds, test_ds = dataset[:t1], dataset[t1:t2], dataset[t2:]
    print(f"  Split → train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # ── Class weights (class balancing) ──
    train_labels = np.array([d.y.item() for d in train_ds])
    counts       = np.bincount(train_labels, minlength=3).astype(np.float32)
    weights      = torch.tensor(1.0 / (counts + 1e-6))
    weights      = (weights / weights.sum() * 3).to(DEVICE)   # normalized to mean=1

    # ── Model ──
    print("\n[3/4] Initializing model...")
    model     = LOB_GAT().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=PATIENCE // 2, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=weights)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

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
            tqdm.write(f"\nEarly stopping at epoch {epoch}.")
            break

    # ── Final evaluation ──
    model.load_state_dict(torch.load("best_lob_gnn.pt", map_location=DEVICE))
    _, test_acc = run_epoch(model, test_loader, None, criterion, train=False)
    print(f"\nTest accuracy: {test_acc:.4f}")
    evaluate_full(model, test_loader)


if __name__ == "__main__":
    main()