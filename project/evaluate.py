"""Evaluation utilities for mapping accuracy."""

from typing import Dict, List, Tuple

import numpy as np


def compute_metrics(errors: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE and RMSE from an array of distance errors.

    Args:
        errors: Array of Euclidean distance errors.

    Returns:
        Dictionary with keys: 'mae', 'rmse'.
    """
    if errors.size == 0:
        return {"mae": 0.0, "rmse": 0.0}

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {"mae": mae, "rmse": rmse}


def evaluate_frame(
    preds: np.ndarray,
    gts: np.ndarray,
    matches: List[Tuple[int, int]],
) -> Dict[str, float]:
    """
    Evaluate a single frame given matched prediction and GT indices.

    Args:
        preds: Predicted points array of shape (N, 2).
        gts: Ground-truth points array of shape (M, 2).
        matches: List of (pred_idx, gt_idx) pairs.

    Returns:
        Dictionary with MAE and RMSE for the frame.
    """
    if not matches:
        return {"mae": 0.0, "rmse": 0.0}

    errors = []
    for pred_idx, gt_idx in matches:
        dist = np.linalg.norm(preds[pred_idx] - gts[gt_idx])
        errors.append(dist)

    errors = np.asarray(errors, dtype=float)
    return compute_metrics(errors)


def summarize_metrics(
    per_frame: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Aggregate per-frame metrics into overall averages.

    Args:
        per_frame: List of metric dicts per frame.

    Returns:
        Dictionary with averaged MAE and RMSE.
    """
    if not per_frame:
        return {"mae": 0.0, "rmse": 0.0}

    mae_vals = [m["mae"] for m in per_frame]
    rmse_vals = [m["rmse"] for m in per_frame]

    return {
        "mae": float(np.mean(mae_vals)),
        "rmse": float(np.mean(rmse_vals)),
    }
