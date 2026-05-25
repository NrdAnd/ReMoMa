import numpy as np

# LOBSTER column indices for best bid/ask (level 0 = best)
_ASK_PRICE_L1 = 0
_BID_PRICE_L1 = 2


def compute_labels(data: np.ndarray, threshold: float, k: int = 1) -> np.ndarray:
    """Compute mid-price direction labels over a k-step horizon.

    Label[t] encodes the direction of mid-price from tick t to tick t+k:
        0  →  down  (pct_change < -threshold)
        1  →  flat  (|pct_change| <= threshold)
        2  →  up    (pct_change >  threshold)

    Args:
        data:      [N, 40] LOBSTER snapshot array.
        threshold: Fractional threshold for flat classification (e.g. 1e-4).
        k:         Prediction horizon in ticks (default 1).

    Returns:
        np.ndarray of shape [N-k], dtype int64.
        Entry at index t is the label for the transition t → t+k.
    """
    mid = (data[:, _ASK_PRICE_L1] + data[:, _BID_PRICE_L1]) / 2.0
    pct = (mid[k:] - mid[:-k]) / (mid[:-k] + 1e-10)

    labels = np.ones(len(pct), dtype=np.int64)
    labels[pct >  threshold] = 2
    labels[pct < -threshold] = 0
    return labels
