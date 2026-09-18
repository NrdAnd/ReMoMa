"""Post-hoc decision-threshold tuning for imbalanced LOB classification.

The trained model emits probabilities for (down, flat, up). On imbalanced LOB
data, argmax can over-predict directional classes because a small directional
edge may beat ``flat`` even when confidence is weak. These helpers keep model
weights fixed and only change the final decision rule: predict down/up only
when the corresponding probability clears a tuned threshold; otherwise predict
flat. Thresholds are tuned on validation and then applied unchanged to test.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


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


def decision_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Metrics focused on the flat-vs-directional trade-off.

    ``signal_*`` treats down/up as directional trading signals and counts a
    signal as correct only when the exact direction is correct.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    pred_signal = preds != 1
    true_signal = labels != 1
    correct_signal = pred_signal & true_signal & (preds == labels)
    true_flat = labels == 1

    n = max(int(labels.size), 1)
    n_pred_signal = int(pred_signal.sum())
    n_true_signal = int(true_signal.sum())
    n_true_flat = int(true_flat.sum())

    signal_precision = (
        float(correct_signal.sum() / n_pred_signal) if n_pred_signal else 0.0
    )
    signal_recall = (
        float(correct_signal.sum() / n_true_signal) if n_true_signal else 0.0
    )
    signal_f1 = (
        2.0 * signal_precision * signal_recall / (signal_precision + signal_recall)
        if signal_precision + signal_recall > 0
        else 0.0
    )
    flat_to_signal_rate = (
        float((true_flat & pred_signal).sum() / n_true_flat) if n_true_flat else 0.0
    )
    opposite_direction = ((labels == 0) & (preds == 2)) | (
        (labels == 2) & (preds == 0)
    )
    opposite_direction_rate = (
        float(opposite_direction.sum() / n_true_signal) if n_true_signal else 0.0
    )

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, labels=[0, 1, 2], average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "signal_precision": signal_precision,
        "signal_recall": signal_recall,
        "signal_f1": signal_f1,
        "signal_rate": float(n_pred_signal / n),
        "true_signal_rate": float(n_true_signal / n),
        "flat_to_signal_rate": flat_to_signal_rate,
        "opposite_direction_rate": opposite_direction_rate,
    }


def _score_metrics(
    metrics: dict[str, float],
    objective: str,
    flat_fp_penalty: float,
) -> float:
    if objective == "macro_f1":
        return metrics["f1_macro"]
    if objective == "macro_f1_penalized":
        return metrics["f1_macro"] - flat_fp_penalty * metrics["flat_to_signal_rate"]
    if objective == "signal_f1":
        return metrics["signal_f1"]
    if objective == "signal_precision":
        return metrics["signal_precision"]
    raise ValueError(
        "objective must be one of: macro_f1, macro_f1_penalized, "
        "signal_f1, signal_precision"
    )


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """Compute fixed-three-class and directional metrics from sufficient counts."""
    cm = np.asarray(cm, dtype=np.float64)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    n = cm.sum()
    if n <= 0:
        raise ValueError("Metrics require at least one sample.")
    diagonal = np.diag(cm)
    denom = support + predicted
    f1 = np.divide(2 * diagonal, denom, out=np.zeros(3), where=denom != 0)
    signal_correct = diagonal[0] + diagonal[2]
    pred_signal, true_signal = predicted[0] + predicted[2], support[0] + support[2]
    precision = float(signal_correct / pred_signal) if pred_signal else 0.0
    recall = float(signal_correct / true_signal) if true_signal else 0.0
    mcc_denom = np.sqrt((n * n - (predicted ** 2).sum()) * (n * n - (support ** 2).sum()))
    return {"accuracy": float(diagonal.sum() / n), "f1_macro": float(f1.mean()),
            "f1_weighted": float(np.dot(f1, support) / n), "f1_per_class": f1.tolist(),
            "mcc": float((diagonal.sum() * n - np.dot(support, predicted)) / mcc_denom) if mcc_denom else 0.0,
            "signal_precision": precision, "signal_recall": recall,
            "signal_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "signal_rate": float(pred_signal / n), "true_signal_rate": float(true_signal / n),
            "flat_to_signal_rate": float((cm[1, 0] + cm[1, 2]) / support[1]) if support[1] else 0.0,
            "opposite_direction_rate": float((cm[0, 2] + cm[2, 0]) / true_signal) if true_signal else 0.0}


def _threshold_confusions(probs, labels, grid):
    """Exact grid counts in O(N + G^2), respecting the up-wins-ties rule."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    if (probs.shape != (len(labels), 3) or not len(labels) or not np.isfinite(probs).all()
            or not np.isin(labels, [0, 1, 2]).all()):
        raise ValueError("Threshold search needs finite [N,3] probabilities and valid labels.")
    g = len(grid)
    # Match NumPy's scalar comparison promotion in apply_thresholds, including
    # probabilities exactly equal to a float32 representation of a grid value.
    comparison_grid = grid.astype(probs.dtype)
    a = np.searchsorted(comparison_grid, probs[:, 0], side="right")
    b = np.searchsorted(comparison_grid, probs[:, 2], side="right")
    up_wins = probs[:, 2] >= probs[:, 0]
    matrices = np.zeros((g, g, 3, 3), dtype=np.int64)
    for label in range(3):
        count = int((labels == label).sum())
        for winner in (False, True):
            selected = (labels == label) & (up_wins == winner)
            hist = np.bincount(a[selected] * (g + 1) + b[selected], minlength=(g + 1) ** 2)
            cumulative = hist.reshape(g + 1, g + 1).cumsum(0).cumsum(1)
            if winner:
                matrices[:, :, label, 2] += cumulative[g, g] - cumulative[g, :g][None, :]
                matrices[:, :, label, 0] += cumulative[g, :g][None, :] - cumulative[:g, :g]
            else:
                matrices[:, :, label, 0] += cumulative[g, g] - cumulative[:g, g][:, None]
                matrices[:, :, label, 2] += cumulative[:g, g][:, None] - cumulative[:g, :g]
        matrices[:, :, label, 1] = count - matrices[:, :, label, 0] - matrices[:, :, label, 2]
    return matrices


def search_thresholds(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    lo: float = 0.33,
    hi: float = 0.90,
    step: float = 0.02,
    objective: str = "macro_f1",
    flat_fp_penalty: float = 0.0,
    min_signal_precision: float | None = None,
    min_signal_recall: float | None = None,
    max_signal_rate: float | None = None,
    max_flat_to_signal_rate: float | None = None,
) -> dict:
    """Grid-search thresholds with optional directional-signal constraints."""
    if not (0 <= lo <= hi <= 1) or step <= 0:
        raise ValueError("Threshold grid requires 0 <= lo <= hi <= 1 and step > 0.")
    if not np.isfinite([lo, hi, step]).all() or step < .001:
        raise ValueError("Threshold search uses 0.001 resolution; set a finite step >= 0.001.")
    grid = np.unique(np.round(np.arange(lo, hi + 1e-9, step), 3))
    matrices = _threshold_confusions(val_probs, val_labels, grid)
    best: dict | None = None

    for i, td in enumerate(grid):
        for j, tu in enumerate(grid):
            # Keep the public threshold-search metric contract scalar-only.
            # Existing fold/ensemble reporters flatten this dictionary.
            metrics = {key: value for key, value in metrics_from_confusion(matrices[i, j]).items()
                       if key not in {"f1_per_class", "mcc"}}

            if (
                min_signal_precision is not None
                and metrics["signal_precision"] < min_signal_precision
            ):
                continue
            if (
                min_signal_recall is not None
                and metrics["signal_recall"] < min_signal_recall
            ):
                continue
            if max_signal_rate is not None and metrics["signal_rate"] > max_signal_rate:
                continue
            if (
                max_flat_to_signal_rate is not None
                and metrics["flat_to_signal_rate"] > max_flat_to_signal_rate
            ):
                continue

            score = _score_metrics(metrics, objective, flat_fp_penalty)
            if best is None or score > best["score"]:
                best = {
                    "t_down": float(td),
                    "t_up": float(tu),
                    "score": float(score),
                    "objective": objective,
                    "metrics": metrics,
                    "search": {
                        "lo": float(lo),
                        "hi": float(hi),
                        "step": float(step),
                        "flat_fp_penalty": float(flat_fp_penalty),
                        "min_signal_precision": min_signal_precision,
                        "min_signal_recall": min_signal_recall,
                        "max_signal_rate": max_signal_rate,
                        "max_flat_to_signal_rate": max_flat_to_signal_rate,
                    },
                }

    if best is None:
        raise ValueError(
            "No threshold pair satisfied the requested constraints. "
            "Relax min/max constraints or widen the search grid."
        )
    return best


def tune_thresholds(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    lo: float = 0.33,
    hi: float = 0.90,
    step: float = 0.02,
    objective: str = "macro_f1",
    flat_fp_penalty: float = 0.0,
    min_signal_precision: float | None = None,
    min_signal_recall: float | None = None,
    max_signal_rate: float | None = None,
    max_flat_to_signal_rate: float | None = None,
) -> tuple[float, float, float]:
    """Backward-compatible threshold tuner.

    Returns:
        (t_down, t_up, best_val_f1_macro)
    """
    result = search_thresholds(
        val_probs,
        val_labels,
        lo=lo,
        hi=hi,
        step=step,
        objective=objective,
        flat_fp_penalty=flat_fp_penalty,
        min_signal_precision=min_signal_precision,
        min_signal_recall=min_signal_recall,
        max_signal_rate=max_signal_rate,
        max_flat_to_signal_rate=max_flat_to_signal_rate,
    )
    return result["t_down"], result["t_up"], result["metrics"]["f1_macro"]
