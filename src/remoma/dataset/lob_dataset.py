import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data, Dataset

from remoma.dataset.binning import VolumeBinner
from remoma.dataset.indexed import FeatureReader

# Column index groups (LOBSTER 40-column format, 0-indexed levels)
_ASK_P_COLS = np.arange(0, 40, 4)   # AskPrice: cols 0,4,8,...,36
_BID_P_COLS = np.arange(2, 40, 4)   # BidPrice: cols 2,6,10,...,38
_ASK_V_COLS = np.arange(1, 40, 4)   # AskVol:   cols 1,5,9,...,37
_BID_V_COLS = np.arange(3, 40, 4)   # BidVol:   cols 3,7,11,...,39


class FastLOBDataset(Dataset):
    """Memory-mapped dataset over precomputed LOB graph node features.

    Expected files:
        X_*.npy: shape [N, 3020, 2] (typically float16 for disk efficiency)
        y_*.npy: shape [N] int64 labels {0,1,2}

    Runtime behavior:
      - x is cast to float32 inside get() for stable training math.
      - edge_index is static and shared by all samples.
      - no online feature reconstruction is performed.
    """

    def __init__(self, x_path: str, y_path: str, edge_index: torch.Tensor):
        super().__init__()
        self.x_path = x_path
        self.y_path = y_path

        x = np.load(self.x_path, mmap_mode="r")
        y = np.load(self.y_path, mmap_mode="r")
        if len(x) != len(y):
            raise ValueError(
                f"Mismatched feature/label lengths: {len(x)} vs {len(y)} "
                f"({x_path}, {y_path})"
            )
        self._len = int(len(y))
        self.X = None
        self.y = None
        self.edge_index = edge_index
        self.reader = FeatureReader(x_path)

    def _ensure_open(self) -> None:
        # Lazily open memmaps per worker process.
        if self.X is None or self.y is None:
            self.X = np.load(self.x_path, mmap_mode="r")
            self.y = np.load(self.y_path, mmap_mode="r")

    def len(self) -> int:
        return self._len

    def get(self, idx: int) -> Data:
        self._ensure_open()
        # float16->float32 cast allocates once; if already float32 on disk,
        # copy only when needed to get a writable NumPy buffer for torch.
        x_np = self.reader.read([idx])[0]
        if not x_np.flags.writeable:
            x_np = x_np.copy()
        x = torch.from_numpy(x_np)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return Data(x=x, edge_index=self.edge_index, y=y)

    def __getitems__(self, indices):
        self._ensure_open()
        features = self.reader.read(indices)
        return [Data(x=torch.from_numpy(x), edge_index=self.edge_index,
                     y=torch.tensor(int(self.y[index]), dtype=torch.long))
                for x, index in zip(features, indices)]

    def __getstate__(self):
        return {**self.__dict__, "X": None, "y": None}


class StaticLOBTensorDataset(TorchDataset):
    """Memory-mapped tensor dataset for static-graph training mode.

    Returns plain tensors:
      - x: float32 tensor of shape [N, F]
      - y: long scalar tensor
    so a standard torch DataLoader builds batches as:
      - x_batch: [B, N, F]
      - y_batch: [B]
    """

    def __init__(self, x_path: str, y_path: str):
        self.x_path = x_path
        self.y_path = y_path

        x = np.load(self.x_path, mmap_mode="r")
        y = np.load(self.y_path, mmap_mode="r")
        if len(x) != len(y):
            raise ValueError(
                f"Mismatched feature/label lengths: {len(x)} vs {len(y)} "
                f"({x_path}, {y_path})"
            )
        self._len = int(len(y))
        self.X = None
        self.y = None
        self.reader = FeatureReader(x_path)

    def _ensure_open(self) -> None:
        # Lazily open memmaps per worker process.
        if self.X is None or self.y is None:
            self.X = np.load(self.x_path, mmap_mode="r")
            self.y = np.load(self.y_path, mmap_mode="r")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        # float16->float32 cast allocates once; if already float32 on disk,
        # copy only when needed to get a writable NumPy buffer for torch.
        x_np = self.reader.read([idx])[0]
        if not x_np.flags.writeable:
            x_np = x_np.copy()
        x = torch.from_numpy(x_np)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y

    def __getitems__(self, indices):
        self._ensure_open()
        features = self.reader.read(indices)
        return [(torch.from_numpy(x), torch.tensor(int(self.y[index]), dtype=torch.long))
                for x, index in zip(features, indices)]

    def __getstate__(self):
        return {**self.__dict__, "X": None, "y": None}


class LOBDataset(Dataset):
    """Lazy-loading PyG Dataset for LOB graph classification.

    Each sample at time t is a graph with 3020 nodes (10 ask levels × 151 lags
    + 10 bid levels × 151 lags). Features are assembled on-the-fly from raw
    LOBSTER arrays so that only ~O(raw_data) memory is required.

    Node ordering matches the TMFG adjacency matrix:
        ask_i_lag_k  →  node index  i*151 + k          (i=0..9, k=0..150)
        bid_i_lag_k  →  node index  1510 + i*151 + k   (i=0..9, k=0..150)

    Args:
        samples:      [N, 2] int32 array of (file_idx, t) pairs.
        file_data:    List of [T_i, 40] float32 arrays (one per file).
        file_labels:  List of [T_i-1] int64 arrays (one per file).
                      labels[fi][t] = direction from tick t to t+1.
        edge_index:   Static [2, E] edge index tensor (same for all samples).
        binner:       Fitted VolumeBinner (fit on training data only).
        price_stats:  Dict with keys ask_mean, ask_std, bid_mean, bid_std,
                      each [10] float32. None disables price normalization.
        n_lags:       Number of lag steps (default 150 → window of 151 ticks).
    """

    def __init__(
        self,
        samples: np.ndarray,
        file_data: list[np.ndarray],
        file_labels: list[np.ndarray],
        edge_index: torch.Tensor,
        binner: VolumeBinner,
        price_stats: dict | None,
        n_lags: int = 150,
    ):
        super().__init__()
        self.samples = samples
        self.file_data = file_data
        self.file_labels = file_labels
        self.edge_index = edge_index
        self.binner = binner
        self.price_stats = price_stats
        self.n_lags = n_lags

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> Data:
        file_idx = int(self.samples[idx, 0])
        t = int(self.samples[idx, 1])

        x = self._build_features(self.file_data[file_idx], t)
        y = torch.tensor(int(self.file_labels[file_idx][t]), dtype=torch.long)

        return Data(x=x, edge_index=self.edge_index, y=y)

    def _build_features(self, data: np.ndarray, t: int) -> torch.Tensor:
        # window_rev[k] = data[t-k] (lag k=0 is most recent, k=n_lags is oldest)
        window_rev = data[t - self.n_lags: t + 1][::-1].copy()  # [151, 40]

        # Shape [n_levels, n_lags+1]: row i, col k → level i at lag k
        ask_prices = window_rev[:, _ASK_P_COLS].T
        bid_prices = window_rev[:, _BID_P_COLS].T
        ask_vols   = window_rev[:, _ASK_V_COLS].T
        bid_vols   = window_rev[:, _BID_V_COLS].T

        if self.price_stats is not None:
            # Normalize prices relative to current mid-price (lag=0).
            # Result is in basis points (×10⁻⁴), scale-free and robust
            # to near-zero variance in deep levels.
            mid = (ask_prices[0, 0] + bid_prices[0, 0]) / 2.0  # scalar
            mid = max(mid, 1.0)
            ask_prices = (ask_prices - mid) / mid
            bid_prices = (bid_prices - mid) / mid

        ask_vols_b, bid_vols_b = self.binner.transform_window(ask_vols, bid_vols)

        # Assemble [3020, 2]: ask nodes first, then bid nodes.
        # ravel() in C order preserves ask_i_lag_k → i*151+k ordering.
        x = np.empty((3020, 2), dtype=np.float32)
        x[:1510, 0] = ask_prices.ravel()
        x[:1510, 1] = ask_vols_b.ravel()
        x[1510:, 0] = bid_prices.ravel()
        x[1510:, 1] = bid_vols_b.ravel()

        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x)
