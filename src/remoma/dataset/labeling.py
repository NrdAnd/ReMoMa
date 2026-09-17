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
    label_mode: str = "pct",
) -> np.ndarray:
    """Compute direction labels over a k-step horizon.

    Label[t] encodes the direction of the reference price from tick t to t+k:
        0  →  down   (score <= -threshold)
        1  →  flat   (-threshold < score < threshold)
        2  →  up     (score >=  threshold)

    Args:
        data:       [N, 40] LOBSTER snapshot array.
        threshold:  Threshold for the flat band. Its meaning depends on label_mode.
        k:          Prediction horizon in ticks (default 1).
        price_type: 'mid' or 'micro'.
        label_mode: 'pct' → score = (price[t+k]-price[t]) / price[t]   (relative)
                          → threshold is a fraction, e.g. 1e-4
                    'abs' → score = price[t+k]-price[t]                 (absolute, price units)
                          → threshold is in price units, e.g. 100 = one $0.01 tick in raw LOBSTER units.
                            More robust: a full-tick move, not any half-tick wiggle.

    Returns:
        np.ndarray of shape [N-k], dtype int64.
    """
    if k < 1 or k >= len(data):
        raise ValueError("Prediction horizon must be positive and shorter than the series.")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative.")
    if price_type not in _PRICE_FN:
        raise ValueError(f"price_type must be 'mid' or 'micro', got '{price_type}'")

    price = _PRICE_FN[price_type](data)
    diff = price[k:] - price[:-k]

    if label_mode == "pct":
        score = diff / (price[:-k] + 1e-10)
    elif label_mode == "abs":
        score = diff
    else:
        raise ValueError(f"label_mode must be 'pct' or 'abs', got '{label_mode}'")

    labels = np.ones(len(score), dtype=np.int64)
    labels[(score >= threshold) & (score > 0)] = 2
    labels[(score <= -threshold) & (score < 0)] = 0
    return labels
