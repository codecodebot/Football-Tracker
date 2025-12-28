"""
Extract marker coordinates from a 2D minimap image.
- Ball: exactly one (yellow)
- Players: multiple (red / blue)
All colors are assumed to be fixed (non-rendered, non-camera).
"""

from typing import List, Tuple
import numpy as np
import cv2


# ===== 반드시 직접 픽셀 찍어서 값 확정할 것 (BGR 기준) =====
BALL_BGR = np.array([0, 230, 255])     # 노란색 공
RED_BGR  = np.array([0, 0, 255])       # 빨간 팀
BLUE_BGR = np.array([255, 0, 0])       # 파란 팀

# 색 거리 허용 임계값 (너무 크면 노이즈 잡힘)
PLAYER_COLOR_THRESH = 10


def _color_distance_map(img: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Compute per-pixel color distance map."""
    return np.linalg.norm(
        img.astype(np.int16) - color.reshape(1, 1, 3),
        axis=2,
    )


# ======================
# 공 좌표 (항상 1개)
# ======================
def extract_ball_position(minimap: np.ndarray) -> Tuple[int, int]:
    """
    Extract the ball position from the minimap.

    Returns:
        (x, y) pixel coordinate
    """
    diff = _color_distance_map(minimap, BALL_BGR)
    y, x = np.unravel_index(np.argmin(diff), diff.shape)
    return int(x), int(y)


# ======================
# 선수 좌표 (여러 개)
# ======================
def extract_player_positions(
    minimap: np.ndarray,
    team: str,
) -> List[Tuple[int, int]]:
    """
    Extract player positions for a given team.

    Args:
        team: "red" or "blue"

    Returns:
        List of (x, y) pixel coordinates
    """
    if team == "red":
        target_color = RED_BGR
    elif team == "blue":
        target_color = BLUE_BGR
    else:
        raise ValueError("team must be 'red' or 'blue'")

    diff = _color_distance_map(minimap, target_color)

    # 색 기준 마스크
    mask = diff < PLAYER_COLOR_THRESH

    # connected component로 마커 묶기
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))

    positions = []
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        cx = int(xs.mean())
        cy = int(ys.mean())
        positions.append((cx, cy))

    return positions
