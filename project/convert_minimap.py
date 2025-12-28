import cv2
import numpy as np
from typing import List, Tuple


RED_BGR   = np.array([55, 40, 165])
BLUE_BGR  = np.array([51, 46, 135])
BALL_BGR  = np.array([53, 189, 234])
WHITE_BGR = np.array([255, 255, 255])

WHITE_THRESH = 70


def color_distance_map(img: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        img.astype(np.int16) - color.reshape(1, 1, 3),
        axis=2,
    )


def is_valid_marker(cnt) -> bool:
    area = cv2.contourArea(cnt)
    if not (8 <= area <= 60):
        return False

    x, y, w, h = cv2.boundingRect(cnt)
    if h == 0:
        return False
    ratio = w / h
    if not (0.85 < ratio < 1.15):
        return False

    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return False

    circularity = 4 * np.pi * area / (peri * peri)
    if circularity < 0.75:
        return False

    return True


def extract_all_marker_centers(minimap: np.ndarray) -> List[Tuple[int, int]]:
    diff = color_distance_map(minimap, WHITE_BGR)
    mask = (diff < WHITE_THRESH).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        if not is_valid_marker(cnt):
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    return centers


def classify_inner_color(minimap: np.ndarray, cx: int, cy: int):
    h, w = minimap.shape[:2]
    r = 2

    x1, x2 = max(0, cx - r), min(w, cx + r + 1)
    y1, y2 = max(0, cy - r), min(h, cy + r + 1)

    patch = minimap[y1:y2, x1:x2].astype(np.int16)
    mean = patch.reshape(-1, 3).mean(axis=0)

    d_ball = np.linalg.norm(mean - BALL_BGR)
    d_red  = np.linalg.norm(mean - RED_BGR)
    d_blue = np.linalg.norm(mean - BLUE_BGR)

    return d_ball, d_red, d_blue


def extract_minimap_entities(minimap: np.ndarray) -> dict:
    centers = extract_all_marker_centers(minimap)

    scored = []
    for cx, cy in centers:
        d_ball, d_red, d_blue = classify_inner_color(minimap, cx, cy)
        scored.append((cx, cy, d_ball, d_red, d_blue))

    if len(scored) == 0:
        return {
            "ball": (0, 0),
            "red": [],
            "blue": [],
        }

    scored.sort(key=lambda x: x[2])
    ball = (scored[0][0], scored[0][1])

    rest = scored[1:]

    red_cands = []
    blue_cands = []

    for cx, cy, _, d_red, d_blue in rest:
        if d_red < d_blue:
            red_cands.append(((cx, cy), d_red))
        else:
            blue_cands.append(((cx, cy), d_blue))

    red_cands.sort(key=lambda x: x[1])
    blue_cands.sort(key=lambda x: x[1])

    red = [p for p, _ in red_cands[:11]]
    blue = [p for p, _ in blue_cands[:11]]

    nr, nb = len(red), len(blue)
    total_detected = nr + nb + 1

    if total_detected == 22:
        if nr == 10:
            red.append(ball)
        elif nb == 10:
            blue.append(ball)

    elif total_detected == 21:
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
