import numpy as np

# LOBSTER column indices for best bid/ask (level 0 = best)
_ASK_PRICE_L1 = 0
_ASK_VOL_L1   = 1
_BID_PRICE_L1 = 2
_BID_VOL_L1   = 3


def _mid_price(data: np.ndarray) -> np.ndarray:
    return (data[:, _ASK_PRICE_L1] + data[:, _BID_PRICE_L1]) / 2.0


def _micro_price(data: np.ndarray) -> np.ndarray:
    ask_p, ask_v = data[:, _ASK_PRICE_L1], data[:, _ASK_VOL_L1]
    bid_p, bid_v = data[:, _BID_PRICE_L1], data[:, _BID_VOL_L1]
    return (ask_p * bid_v + bid_p * ask_v) / (ask_v + bid_v + 1e-10)


_PRICE_FN = {"mid": _mid_price, "micro": _micro_price}


def compute_labels(
    data: np.ndarray,
    threshold: float,
    k: int = 1,
    price_type: str = "mid",
) -> np.ndarray:
    """Compute direction labels over a k-step horizon.

    Label[t] encodes the direction of the reference price from tick t to t+k:
        0  →  down  (pct_change < -threshold)
        1  →  flat  (|pct_change| <= threshold)
        2  →  up    (pct_change >  threshold)

    Args:
        data:       [N, 40] LOBSTER snapshot array.
        threshold:  Fractional threshold for flat (e.g. 1e-4).
                    Use 0 with price_type='micro' for balanced classes.
        k:          Prediction horizon in ticks (default 1).
        price_type: 'mid'   → (AskPrice_L1 + BidPrice_L1) / 2
                    'micro' → volume-weighted mid-price (order imbalance aware)

    Returns:
        np.ndarray of shape [N-k], dtype int64.
    """
    if price_type not in _PRICE_FN:
        raise ValueError(f"price_type must be 'mid' or 'micro', got '{price_type}'")

    price = _PRICE_FN[price_type](data)
    pct = (price[k:] - price[:-k]) / (price[:-k] + 1e-10)

    labels = np.ones(len(pct), dtype=np.int64)
    labels[pct >  threshold] = 2
    labels[pct < -threshold] = 0
    return labels
