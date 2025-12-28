import cv2
import numpy as np
from typing import List, Tuple


# ===== 기준 색 (BGR) =====
RED_INNER_BGR  = np.array([55, 40, 165])
BLUE_INNER_BGR = np.array([51, 46, 135])
BALL_BGR       = np.array([53, 189, 234])

MIN_RING_AREA = 5
INNER_COLOR_THRESH = 80


def color_distance(img: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        img.astype(np.int16) - color.reshape(1, 1, 3),
        axis=2,
    )


# ===== 1. 링(도넛) 형태 검출 =====
def extract_ring_centers(minimap: np.ndarray) -> List[Tuple[int, int]]:
    gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return []

    centers = []
    hierarchy = hierarchy[0]

    for i, cnt in enumerate(contours):
        child = hierarchy[i][2]
        parent = hierarchy[i][3]

        if child == -1 or parent != -1:
            continue

        area = cv2.contourArea(cnt)
        if area < MIN_RING_AREA:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    return centers


# ===== 2. 링 내부 색 샘플링 =====
def sample_inner_color(minimap: np.ndarray, cx: int, cy: int) -> np.ndarray:
    h, w = minimap.shape[:2]
    r = 2

    x1, x2 = max(0, cx - r), min(w, cx + r + 1)
    y1, y2 = max(0, cy - r), min(h, cy + r + 1)

    patch = minimap[y1:y2, x1:x2].astype(np.int16)
    return patch.reshape(-1, 3).mean(axis=0)


# ===== 3. 공: 가장 노란 픽셀 =====
def extract_ball_position(minimap: np.ndarray) -> Tuple[int, int]:
    diff = color_distance(minimap, BALL_BGR)
    y, x = np.unravel_index(np.argmin(diff), diff.shape)
    return int(x), int(y)


# ===== 4. 메인 로직 =====
def extract_minimap_entities(minimap: np.ndarray) -> dict:
    ball = extract_ball_position(minimap)

    ring_centers = extract_ring_centers(minimap)

    red_scores = []
    blue_scores = []

    for cx, cy in ring_centers:
        mean_color = sample_inner_color(minimap, cx, cy)

        d_red = np.linalg.norm(mean_color - RED_INNER_BGR)
        d_blue = np.linalg.norm(mean_color - BLUE_INNER_BGR)

        red_scores.append(((cx, cy), d_red))
        blue_scores.append(((cx, cy), d_blue))

    red_scores.sort(key=lambda x: x[1])
    blue_scores.sort(key=lambda x: x[1])

    red = [p for p, _ in red_scores[:11]]
    blue = [p for p, _ in blue_scores[:11]]

    nr, nb = len(red), len(blue)
    total = nr + nb + 1

    # ===== 겹침 보정 로직 =====
    if total == 22:
        if nr == 10:
            red.append(ball)
        elif nb == 10:
            blue.append(ball)

    elif total == 21:
        if nr == 10 and nb == 10:
            red.append(ball)
            blue.append(ball)

    red = red[:11]
    blue = blue[:11]

    return {
        "ball": ball,
        "red": red,
        "blue": blue,
    }
