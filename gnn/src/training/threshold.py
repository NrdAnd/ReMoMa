"""Post-hoc decision-threshold tuning for imbalanced F1 macro.

A model trained on balanced classes over-predicts the rare classes when the
eval set is imbalanced. These helpers keep the model fixed and only change the
DECISION RULE: predict down/up only when confident enough (prob >= threshold),
else flat. Thresholds are tuned on validation and applied unchanged to test.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def apply_thresholds(probs: np.ndarray, t_down: float, t_up: float) -> np.ndarray:
    """Predict up/down only if confident enough, else flat.

    Args:
        probs: [N, 3] class probabilities, columns ordered (down, flat, up).
        t_down, t_up: minimum probability required to predict down / up.
    Returns:
        [N] int64 predictions in {0,1,2}.
    """
    preds = np.ones(len(probs), dtype=np.int64)            # default flat
    up = probs[:, 2] >= t_up
    dn = probs[:, 0] >= t_down
    preds[up & ~dn] = 2
    preds[dn & ~up] = 0
    both = up & dn
    preds[both] = np.where(probs[both, 2] >= probs[both, 0], 2, 0)
    return preds


def tune_thresholds(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    lo: float = 0.33,
    hi: float = 0.90,
    step: float = 0.02,
) -> tuple[float, float, float]:
    """Grid-search (t_down, t_up) maximizing F1 macro on the validation set.

    Returns:
        (t_down, t_up, best_val_f1_macro)
    """
    grid = np.round(np.arange(lo, hi + 1e-9, step), 3)
    best_td, best_tu, best_f1 = 0.33, 0.33, -1.0
    for td in grid:
        for tu in grid:
            f1 = f1_score(
                val_labels, apply_thresholds(val_probs, td, tu),
                average="macro", zero_division=0,
            )
            if f1 > best_f1:
                best_td, best_tu, best_f1 = float(td), float(tu), float(f1)
    return best_td, best_tu, best_f1
