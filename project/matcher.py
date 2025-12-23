"""Matching utilities using the Hungarian algorithm."""

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_match(
    preds: np.ndarray,
    gts: np.ndarray,
    max_distance: float = 200.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match predicted points to ground-truth points using Hungarian assignment.

    Args:
        preds: Predicted points array of shape (N, 2).
        gts: Ground-truth points array of shape (M, 2).
        max_distance: Maximum allowed distance for a valid match.

    Returns:
        matches: List of (pred_idx, gt_idx) matched pairs.
        unmatched_preds: List of indices of unmatched predictions.
        unmatched_gts: List of indices of unmatched ground truths.
    """
    if len(preds) == 0 or len(gts) == 0:
        return [], list(range(len(preds))), list(range(len(gts)))

    # Cost matrix: pairwise Euclidean distances
    cost = np.linalg.norm(
        preds[:, None, :] - gts[None, :, :],
        axis=2,
    )

    row_idx, col_idx = linear_sum_assignment(cost)

    matches: List[Tuple[int, int]] = []
    unmatched_preds = set(range(len(preds)))
    unmatched_gts = set(range(len(gts)))

    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= max_distance:
            matches.append((r, c))
            unmatched_preds.discard(r)
            unmatched_gts.discard(c)

    return matches, sorted(unmatched_preds), sorted(unmatched_gts)
