from __future__ import annotations

import numpy as np
import pandas as pd

# LOBSTER message columns: time, event_type, order_id, size, price, direction
# event_type: 1=new limit, 2=partial cancel, 3=delete, 4=visible execution,
#             5=hidden execution, 6=cross, 7=halt
# direction:  1=buy/bid limit order, -1=sell/ask limit order


def load_message_csv(path: str) -> np.ndarray:
    """Load a LOBSTER message file → [N, 6] float64 (time,type,id,size,price,dir)."""
    values = pd.read_csv(path, header=None, usecols=[0, 1, 2, 3, 4, 5],
                       dtype=np.float64, on_bad_lines="error").values
    if not np.isfinite(values).all():
        raise ValueError(f"Message values must be finite: {path}.")
    return values


def per_event_order_flow(msg: np.ndarray) -> np.ndarray:
    """Per-event signed order-flow signals, row-aligned with the orderbook.

    For each message row returns 3 channels (buy-positive convention):
        [0] trade_flow : aggressive-trade signed volume.
            type 4/5 execution with limit-dir d is hit by aggressor of side -d,
            so buy pressure = size * (-d).
        [1] limit_flow : liquidity added by new limit orders = size * d (type 1).
        [2] cancel_flow: liquidity removed by cancels = -size * d (type 2/3).

    Returns: [N, 3] float32.
    """
    etype = msg[:, 1].astype(np.int64)
    size = msg[:, 3].astype(np.float64)
    d = msg[:, 5].astype(np.float64)             # 1 bid, -1 ask

    trade = np.where(np.isin(etype, (4, 5)), size * (-d), 0.0)
    limit = np.where(etype == 1, size * d, 0.0)
    cancel = np.where(np.isin(etype, (2, 3)), -size * d, 0.0)

    return np.stack([trade, limit, cancel], axis=1).astype(np.float32)


def windowed_order_flow(of: np.ndarray, t: int, n_lags: int) -> np.ndarray:
    """Aggregate per-event order flow over the lag window [t-n_lags, t].

    Returns a compact feature vector summarising the recent order flow at tick t:
        sum of trade/limit/cancel flow over the window (net pressure),
        plus the most-recent-tick trade flow (instantaneous aggressor).
    """
    w = of[t - n_lags: t + 1]                    # [n_lags+1, 3]
    net = w.sum(axis=0)                           # [3] cumulative over window
    last = of[t]                                  # [3] instantaneous
    return np.concatenate([net, last]).astype(np.float32)   # [6]
