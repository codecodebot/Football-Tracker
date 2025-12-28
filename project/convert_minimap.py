import cv2
import numpy as np
from typing import List, Tuple


# =========================
# 파라미터
# =========================
MIN_RING_AREA = 5

# HSV 기준 (OpenCV: H 0~179)
BLUE_HUE_CENTER = 110        # 파랑 중심
RED_HUE_1 = 0                # 빨강 양끝
RED_HUE_2 = 179

# 빨강 채도 기준(낮으면 패널티)
RED_MIN_SAT = 60
RED_SAT_PENALTY_W = 0.8

# 공 (노란색, BGR)
BALL_BGR = np.array([53, 189, 234])


# =========================
# 유틸
# =========================
def color_distance(img: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        img.astype(np.int16) - color.reshape(1, 1, 3),
        axis=2,
    )


def hue_distance(h: float, center: float) -> float:
    d = abs(h - center)
    return min(d, 180 - d)


# =========================
# 1. 링(도넛) 형태 검출
# =========================
def extract_ring_centers(minimap: np.ndarray) -> List[Tuple[int, int]]:
    # 밝은 테두리 기반 이진화
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

        # 부모이면서 자식(구멍)을 가진 컨투어만 = 링
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


# =========================
# 2. 링 내부 HSV 샘플
# =========================
def sample_inner_hsv(minimap: np.ndarray, cx: int, cy: int) -> Tuple[float, float, float]:
    h, w = minimap.shape[:2]
    r = 2

    x1, x2 = max(0, cx - r), min(w, cx + r + 1)
    y1, y2 = max(0, cy - r), min(h, cy + r + 1)

    patch = minimap[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean = hsv.reshape(-1, 3).mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


# =========================
# 3. 공: 가장 노란 픽셀
# =========================
def extract_ball_position(minimap: np.ndarray) -> Tuple[int, int]:
    diff = color_distance(minimap, BALL_BGR)
    y, x = np.unravel_index(np.argmin(diff), diff.shape)
    return int(x), int(y)


# =========================
# 4. 점수 함수 (HSV)
# =========================
def blue_score(h: float, s: float) -> float:
    # 파랑은 Hue만으로도 안정
    return hue_distance(h, BLUE_HUE_CENTER)


def red_score(h: float, s: float) -> float:
    # 빨강은 Hue + 채도 패널티
    d_h = min(hue_distance(h, RED_HUE_1), hue_distance(h, RED_HUE_2))
    sat_penalty = max(0.0, RED_MIN_SAT - s) * RED_SAT_PENALTY_W
    return d_h + sat_penalty


# =========================
# 5. 메인
# =========================
def extract_minimap_entities(minimap: np.ndarray) -> dict:
    # 공
    ball = extract_ball_position(minimap)

    # 링 검출
    ring_centers = extract_ring_centers(minimap)

    # HSV 점수 계산
    blue_scores = []
    red_scores = []
    for cx, cy in ring_centers:
        h, s, v = sample_inner_hsv(minimap, cx, cy)
        blue_scores.append(((cx, cy), blue_score(h, s)))
        red_scores.append(((cx, cy), red_score(h, s)))

    # 1) 파랑 anchor 먼저 선택
    blue_scores.sort(key=lambda x: x[1])
    blue = [p for p, _ in blue_scores[:11]]
    blue_set = set(blue)

    # 2) 파랑으로 뽑히지 않은 나머지에서 빨강 선택
    red_candidates = [(p, sc) for (p, sc) in red_scores if p not in blue_set]
    red_candidates.sort(key=lambda x: x[1])
    red = [p for p, _ in red_candidates[:11]]

    # 3) 겹침 보정 (21/22/23 규칙)
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

    # 안전 클립
    red = red[:11]
    blue = blue[:11]

    return {
        "ball": ball,
        "red": red,
        "blue": blue,
    }
