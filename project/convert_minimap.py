import cv2
import numpy as np
from typing import List, Tuple


RED_RING_BGR  = np.array([157, 178, 195])   # rgba(195,178,157)
BLUE_RING_BGR = np.array([163, 164, 145])   # rgba(145,164,163)
BALL_BGR      = np.array([53, 189, 234])    # yellow ball

RING_THRESH = 35


def color_distance(img: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        img.astype(np.int16) - color.reshape(1, 1, 3),
        axis=2,
    )


def extract_ring_centers(minimap: np.ndarray, ring_color: np.ndarray) -> List[Tuple[int, int]]:
    diff = color_distance(minimap, ring_color)
    mask = (diff < RING_THRESH).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    return centers


def extract_ball_position(minimap: np.ndarray) -> Tuple[int, int]:
    diff = color_distance(minimap, BALL_BGR)
    y, x = np.unravel_index(np.argmin(diff), diff.shape)
    return int(x), int(y)


def extract_minimap_entities(minimap: np.ndarray) -> dict:
    ball = extract_ball_position(minimap)

    red = extract_ring_centers(minimap, RED_RING_BGR)
    blue = extract_ring_centers(minimap, BLUE_RING_BGR)

    nr, nb = len(red), len(blue)
    total = nr + nb + 1

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
