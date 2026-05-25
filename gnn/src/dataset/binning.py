from __future__ import annotations

import pickle
import numpy as np
from pathlib import Path

# LOBSTER column indices for volume per level (0-indexed levels)
_ASK_VOL_COLS = np.arange(1, 40, 4)   # [1, 5, 9, ..., 37]  AskVol_L1..L10
_BID_VOL_COLS = np.arange(3, 40, 4)   # [3, 7, 11, ..., 39]  BidVol_L1..L10


class VolumeBinner:
    """Quantile-based discretizer for LOB volume columns.

    Fits 20 independent binners: one per (side, level) pair.
    Outputs normalized bin indices in [0, 1] as float32.

    Bin edges are computed on training data only — must call fit() before
    transform_window(), and only on training split data to avoid leakage.
    """

    def __init__(self, n_bins: int = 2000):
        self.n_bins = n_bins
        self._ask_edges: list[np.ndarray] | None = None   # 10 arrays
        self._bid_edges: list[np.ndarray] | None = None   # 10 arrays

    def fit(self, data_list: list[np.ndarray]) -> "VolumeBinner":
        """Fit quantile edges on training data.

        Args:
            data_list: list of [N_i, 40] float32 arrays (training files only).
        """
        all_data = np.concatenate(data_list, axis=0)
        percentiles = np.linspace(0, 100, self.n_bins + 1)

        self._ask_edges = [
            np.unique(np.percentile(all_data[:, col].astype(np.float64), percentiles))
            for col in _ASK_VOL_COLS
        ]
        self._bid_edges = [
            np.unique(np.percentile(all_data[:, col].astype(np.float64), percentiles))
            for col in _BID_VOL_COLS
        ]
        return self

    def transform_window(
        self,
        ask_vols: np.ndarray,
        bid_vols: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bin raw volumes for a single sample window.

        Args:
            ask_vols: [n_levels, n_lags+1] raw ask volumes.
            bid_vols: [n_levels, n_lags+1] raw bid volumes.

        Returns:
            Tuple of two [n_levels, n_lags+1] float32 arrays, values in [0, 1].
        """
        ask_out = np.empty_like(ask_vols, dtype=np.float32)
        bid_out = np.empty_like(bid_vols, dtype=np.float32)

        for i in range(ask_vols.shape[0]):
            ask_out[i] = self._discretize(ask_vols[i], self._ask_edges[i])
            bid_out[i] = self._discretize(bid_vols[i], self._bid_edges[i])

        return ask_out, bid_out

    @staticmethod
    def _discretize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        n_bins = len(edges) - 1
        idx = np.searchsorted(edges[1:-1], values, side="right").astype(np.float32)
        return idx / n_bins

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"n_bins": self.n_bins, "ask": self._ask_edges, "bid": self._bid_edges},
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "VolumeBinner":
        with open(path, "rb") as f:
            state = pickle.load(f)
        binner = cls(n_bins=state["n_bins"])
        binner._ask_edges = state["ask"]
        binner._bid_edges = state["bid"]
        return binner
