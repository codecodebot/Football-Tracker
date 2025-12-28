from typing import List, Tuple
import numpy as np
import cv2


RED_BGR  = np.array([55, 40, 165])
BLUE_BGR = np.array([51, 46, 135])
BALL_BGR = np.array([53, 189, 234])

PLAYER_COLOR_THRESH = 30
MIN_MARKER_AREA = 12


def color_distance_map(img: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        img.astype(np.int16) - color.reshape(1, 1, 3),
        axis=2,
    )


def is_valid_marker(cnt) -> bool:
    area = cv2.contourArea(cnt)
    if area < MIN_MARKER_AREA:
        return False

    x, y, w, h = cv2.boundingRect(cnt)
    ratio = w / h if h > 0 else 0
    if not (0.7 < ratio < 1.3):
        return False

    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return False

    circularity = 4 * np.pi * area / (peri * peri)
    if circularity < 0.6:
        return False

    return True


def extract_ball_position(minimap: np.ndarray) -> Tuple[int, int]:
    diff = color_distance_map(minimap, BALL_BGR)
    y, x = np.unravel_index(np.argmin(diff), diff.shape)
    return int(x), int(y)


def extract_player_positions(
    minimap: np.ndarray,
    team: str,
    max_players: int = 11,
) -> List[Tuple[int, int]]:
    if team == "red":
        target_color = RED_BGR
    elif team == "blue":
        target_color = BLUE_BGR
    else:
        raise ValueError("team must be 'red' or 'blue'")

    diff = color_distance_map(minimap, target_color)
    mask = (diff < PLAYER_COLOR_THRESH).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        if not is_valid_marker(cnt):
            continue

        ys, xs = cnt[:, 0, 1], cnt[:, 0, 0]
        cx = int(xs.mean())
        cy = int(ys.mean())
        color_dist = diff[cy, cx]
        candidates.append(((cx, cy), color_dist))

    candidates.sort(key=lambda x: x[1])
    return [pos for pos, _ in candidates[:max_players]]


def extract_minimap_entities(minimap: np.ndarray) -> dict:
    return {
        "ball": extract_ball_position(minimap),
        "red": extract_player_positions(minimap, "red", 11),
        "blue": extract_player_positions(minimap, "blue", 11),
    }
