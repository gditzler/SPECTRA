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
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true_arr, y_pred_arr):
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
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    if len(y_true_arr) == 0:
        return 0.0
    return float(np.mean(y_true_arr == y_pred_arr))


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
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    classes = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
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
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    snr_values_arr = np.asarray(snr_values, dtype=float)

    result = {}
    for snr in np.unique(snr_values_arr):
        mask = snr_values_arr == snr
        if mask.sum() > 0:
            result[float(snr)] = float(np.mean(y_true_arr[mask] == y_pred_arr[mask]))
    return result


def per_snr_rmse(
    true_angles: Sequence[float],
    estimated_angles: Sequence[float],
    snr_values: Sequence[float],
) -> Dict[float, float]:
    """Compute angular RMSE in **degrees** grouped by SNR bin.

    Args:
        true_angles: Ground-truth azimuth angles in radians.
        estimated_angles: Estimated azimuth angles in radians.
        snr_values: SNR value (dB) for each sample.

    Returns:
        Dict mapping each unique SNR value to the RMSE in degrees at that SNR.
    """
    true_angles_arr = np.asarray(true_angles, dtype=float)
    estimated_angles_arr = np.asarray(estimated_angles, dtype=float)
    snr_values_arr = np.asarray(snr_values, dtype=float)

    if len(true_angles_arr) == 0:
        return {}

    result: Dict[float, float] = {}
    for snr in np.unique(snr_values_arr):
        mask = snr_values_arr == snr
        if mask.sum() > 0:
            errors = true_angles_arr[mask] - estimated_angles_arr[mask]
            rmse_rad = float(np.sqrt(np.mean(errors ** 2)))
            result[float(snr)] = float(np.rad2deg(rmse_rad))
    return result


def bit_error_rate(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Fraction of bits that differ between transmitted and received sequences."""
    tx_bits_arr = np.asarray(tx_bits)
    rx_bits_arr = np.asarray(rx_bits)
    return float(np.mean(tx_bits_arr != rx_bits_arr))


def symbol_error_rate(tx_indices: np.ndarray, rx_indices: np.ndarray) -> float:
    """Fraction of symbol indices that differ."""
    tx_indices_arr = np.asarray(tx_indices)
    rx_indices_arr = np.asarray(rx_indices)
    return float(np.mean(tx_indices_arr != rx_indices_arr))


def packet_error_rate(
    tx_bits: np.ndarray, rx_bits: np.ndarray, packet_length: int
) -> float:
    """Fraction of fixed-length packets containing at least one bit error.

    Truncates to the largest multiple of ``packet_length``.
    """
    tx_bits_arr = np.asarray(tx_bits)
    rx_bits_arr = np.asarray(rx_bits)
    n_packets = len(tx_bits_arr) // packet_length
    if n_packets == 0:
        return 0.0
    usable = n_packets * packet_length
    tx_blocks = tx_bits_arr[:usable].reshape(n_packets, packet_length)
    rx_blocks = rx_bits_arr[:usable].reshape(n_packets, packet_length)
    packet_errors = np.any(tx_blocks != rx_blocks, axis=1)
    return float(np.mean(packet_errors))
