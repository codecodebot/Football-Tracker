import cv2
import numpy as np
from typing import List, Tuple


# ===== HSV 기준 =====
BLUE_HUE_CENTER = 110   # 파랑 중심
RED_HUE_CENTER_1 = 0
RED_HUE_CENTER_2 = 179

MIN_RING_AREA = 5


# ===== 공 (노란색, BGR) =====
BALL_BGR = np.array([53, 189, 234])


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


# ===== 2. 링 내부 HSV 샘플 =====
def sample_inner_hsv(minimap: np.ndarray, cx: int, cy: int) -> np.ndarray:
    h, w = minimap.shape[:2]
    r = 2

    x1, x2 = max(0, cx - r), min(w, cx + r + 1)
    y1, y2 = max(0, cy - r), min(h, cy + r + 1)

    patch = minimap[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    return hsv.reshape(-1, 3).mean(axis=0)


def hue_distance(h, center):
    return min(abs(h - center), 180 - abs(h - center))


# ===== 3. 공: 가장 노란 픽셀 =====
def extract_ball_position(minimap: np.ndarray) -> Tuple[int, int]:
    diff = color_distance(minimap, BALL_BGR)
    y, x = np.unravel_index(np.argmin(diff), diff.shape)
    return int(x), int(y)


# ===== 4. 메인 로직 =====
def extract_minimap_entities(minimap: np.ndarray) -> dict:
    ball = extract_ball_position(minimap)
    ring_centers = extract_ring_centers(minimap)

    blue_scores = []
    red_scores = []

    for cx, cy in ring_centers:
        h, s, v = sample_inner_hsv(minimap, cx, cy)

        d_blue = hue_distance(h, BLUE_HUE_CENTER)
        d_red = min(
            hue_distance(h, RED_HUE_CENTER_1),
            hue_distance(h, RED_HUE_CENTER_2),
        )

        blue_scores.append(((cx, cy), d_blue))
        red_scores.append(((cx, cy), d_red))

    blue_scores.sort(key=lambda x: x[1])
    red_scores.sort(key=lambda x: x[1])

    blue = [p for p, _ in blue_scores[:11]]
    red = [p for p, _ in red_scores[:11]]

    nr, nb = len(red), len(blue)
    total = nr + nb + 1

    # ===== 겹침 보정 =====
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
