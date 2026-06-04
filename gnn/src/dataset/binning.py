from __future__ import annotations

import pickle
import numpy as np
from pathlib import Path

# LOBSTER column indices for volume per level (0-indexed levels)
_ASK_VOL_COLS = np.arange(1, 40, 4)   # [1, 5, 9, ..., 37]  AskVol_L1..L10
_BID_VOL_COLS = np.arange(3, 40, 4)   # [3, 7, 11, ..., 39]  BidVol_L1..L10


class VolumeBinner:
    """Quantile-based discretizer for LOB volume columns.

    Fits a single global binner on all volume values across all levels and
    sides. Outputs normalized bin indices in [0, 1] as float32.

    Bin edges are computed on training data only — must call fit() before
    transform_window(), and only on training split data to avoid leakage.
    """

    def __init__(self, n_bins: int = 2000):
        self.n_bins = n_bins
        self._edges: np.ndarray | None = None   # single shared bin edge array

    def fit(self, data_list: list[np.ndarray]) -> "VolumeBinner":
        """Fit global quantile edges on all volume columns of training data.

        Args:
            data_list: list of [N_i, 40] float32 arrays (training files only).
        """
        all_data = np.concatenate(data_list, axis=0)
        all_vol_cols = np.concatenate([_ASK_VOL_COLS, _BID_VOL_COLS])
        all_vols = all_data[:, all_vol_cols].astype(np.float64).ravel()

        # Drop NaN/inf: a single non-finite value poisons np.percentile and
        # collapses the edges to [nan], silently zeroing the whole feature.
        all_vols = all_vols[np.isfinite(all_vols)]
        if all_vols.size == 0:
            raise ValueError("No finite volume values to fit VolumeBinner.")

        percentiles = np.linspace(0, 100, self.n_bins + 1)
        self._edges = np.unique(np.percentile(all_vols, percentiles))
        if len(self._edges) < 2:
            raise ValueError(
                f"VolumeBinner produced {len(self._edges)} edge(s); volume data "
                "is constant or degenerate. Check the raw input."
            )
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
        return (
            self._discretize(ask_vols),
            self._discretize(bid_vols),
        )

    def _discretize(self, values: np.ndarray) -> np.ndarray:
        if self._edges is None:
            raise RuntimeError("VolumeBinner is not fitted. Call fit() or load() before transform.")
        n_bins = max(len(self._edges) - 1, 1)
        # Map non-finite volumes to 0 (lowest bin) so searchsorted stays valid.
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.searchsorted(self._edges[1:-1], values, side="right").astype(np.float32)
        return idx / n_bins

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"n_bins": self.n_bins, "edges": self._edges}, f)

    @classmethod
    def load(cls, path: str | Path) -> "VolumeBinner":
        with open(path, "rb") as f:
            state = pickle.load(f)
        binner = cls(n_bins=state["n_bins"])
        binner._edges = state["edges"]
        return binner
