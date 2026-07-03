import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

_CLASS_NAMES = ["down", "flat", "up"]


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "accuracy":    accuracy_score(labels, preds),
        "f1_macro":    f1_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "f1_per_class": f1_score(labels, preds, average=None,      zero_division=0).tolist(),
        "mcc":         matthews_corrcoef(labels, preds),
    }


def format_report(preds: np.ndarray, labels: np.ndarray) -> str:
    rep = classification_report(labels, preds, target_names=_CLASS_NAMES, digits=4)
    cm = confusion_matrix(labels, preds)
    return f"{rep}\nConfusion Matrix:\n{cm}"


def print_report(preds: np.ndarray, labels: np.ndarray) -> None:
    print(format_report(preds, labels))
