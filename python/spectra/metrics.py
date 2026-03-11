from typing import Dict, List, Optional, Sequence

import numpy as np


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
        num_classes: Total number of classes.

    Returns:
        [num_classes, num_classes] array where entry (i, j) is the count
        of samples with true label i predicted as j.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute overall classification accuracy.

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.

    Returns:
        Fraction of correctly classified samples.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, F1, and support.

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
        class_names: Optional list of class name strings.

    Returns:
        Dict mapping class name to {"precision", "recall", "f1", "support"}.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = int(classes.max()) + 1 if len(classes) > 0 else 0

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    cm = confusion_matrix(y_true, y_pred, num_classes)
    report = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        name = class_names[i] if i < len(class_names) else str(i)
        report[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return report


def per_snr_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    snr_values: Sequence[float],
) -> Dict[float, float]:
    """Compute accuracy grouped by SNR bin.

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
        snr_values: SNR value for each sample.

    Returns:
        Dict mapping SNR value to accuracy at that SNR.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    snr_values = np.asarray(snr_values, dtype=float)

    result = {}
    for snr in np.unique(snr_values):
        mask = snr_values == snr
        if mask.sum() > 0:
            result[float(snr)] = float(np.mean(y_true[mask] == y_pred[mask]))
    return result
