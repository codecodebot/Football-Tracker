"""Projection utilities for mapping minimap coordinates to a field map."""

from typing import Tuple, List

import cv2
import numpy as np


def create_field_map(width: int, height: int) -> np.ndarray:
    """
    Create a simple top-down soccer field visualization.

    Args:
        width: Field map width.
        height: Field map height.

    Returns:
        BGR image of the field map.
    """
    field = np.zeros((height, width, 3), dtype=np.uint8)
    field[:, :] = (30, 120, 30)

    # Outer boundary
    cv2.rectangle(
        field,
        (5, 5),
        (width - 5, height - 5),
        (255, 255, 255),
        2,
    )

    # Center line
    cv2.line(
        field,
        (width // 2, 5),
        (width // 2, height - 5),
        (255, 255, 255),
        2,
    )

    return field


def compute_homography(
    minimap_size: Tuple[int, int],
    field_size: Tuple[int, int],
) -> np.ndarray:
    """
    Compute homography mapping minimap coordinates to field-map coordinates.

    Args:
        minimap_size: (width, height) of the minimap.
        field_size: (width, height) of the field map.

    Returns:
        3x3 homography matrix.
    """
    w, h = minimap_size
    fw, fh = field_size

    src = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]],
        dtype=np.float32,
    )
    dst = np.array(
        [[0, 0], [fw, 0], [fw, fh], [0, fh]],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(src, dst)
    return H


def project_points(
    points: List[Tuple[float, float]],
    homography: np.ndarray,
) -> np.ndarray:
    """
    Project minimap points onto the field map using a homography.

    Args:
        points: List of (x, y) minimap coordinates.
        homography: 3x3 homography matrix.

    Returns:
        Array of projected points with shape (N, 2).
    """
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, homography)
    return projected.reshape(-1, 2)
