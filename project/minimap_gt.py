"""Extract ground-truth player positions from the minimap."""

from typing import List, Tuple

import cv2
import numpy as np


def extract_minimap_points(
    frame: np.ndarray,
    minimap_roi: Tuple[int, int, int, int],
    hsv_lower: Tuple[int, int, int],
    hsv_upper: Tuple[int, int, int],
) -> List[Tuple[float, float]]:
    """
    Extract player positions from the in-game minimap using HSV thresholding.

    Args:
        frame: Full BGR frame.
        minimap_roi: (x, y, w, h) region of the minimap in the frame.
        hsv_lower: Lower HSV bound for player dots.
        hsv_upper: Upper HSV bound for player dots.

    Returns:
        List of (x, y) coordinates in minimap-local space.
    """
    x, y, w, h = minimap_roi
    minimap = frame[y : y + h, x : x + w]

    if minimap.size == 0:
        return []

    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # Noise reduction
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    points: List[Tuple[float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 4:
            continue

        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue

        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        points.append((cx, cy))

    return points
