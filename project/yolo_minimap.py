import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# 1. YOLO 기반 마커 위치 후보 검출
# ============================================================

class YOLOMarkerDetector:
    def __init__(self, model_path="minimap_marker.pt", device="cpu"):
        self.model = YOLO(model_path)
        self.model.to(device)

    def detect(self, frame, conf_thres=0.3):
        results = self.model.predict(
            frame,
            conf=conf_thres,
            verbose=False
        )[0]

        centers = []

        if results.boxes is None:
            return centers

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centers.append((cx, cy))

        return centers


# ============================================================
# 2. HSV 기반 팀 분류
# ============================================================

def classify_teams(frame, centers):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    scored = []

    for (x, y) in centers:
        if y < 0 or y >= frame.shape[0] or x < 0 or x >= frame.shape[1]:
            continue

        h, s, v = hsv[y, x]

        # 🔵 파랑 (anchor)
        blue_score = -abs(int(h) - 110)

        # 🔴 빨강 (0 / 179 근처)
        red_dist = min(abs(int(h) - 0), abs(int(h) - 179))
        red_score = -red_dist + 0.2 * s

        scored.append({
            "pos": (x, y),
            "blue": blue_score,
            "red": red_score
        })

    # 파랑 11명 먼저 고정
    scored.sort(key=lambda x: x["blue"], reverse=True)
    blue = [p["pos"] for p in scored[:11]]

    # 나머지 중 빨강 상위 11
    rest = scored[11:]
    rest.sort(key=lambda x: x["red"], reverse=True)
    red = [p["pos"] for p in rest[:11]]

    return blue, red


# ============================================================
# 3. 공(ball) 검출 – 가장 노란 픽셀 1개
# ============================================================

def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([20, 120, 120])
    upper = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    # 가장 강한 노란 픽셀 하나
    idx = np.argmax(xs)
    return (int(xs[idx]), int(ys[idx]))


# ============================================================
# 4. 겹침(overlap) 보정
# ============================================================

def fix_overlap(blue, red, ball):
    if ball is None:
        return blue[:11], red[:11], ball

    if len(blue) < 11:
        blue = blue + [ball] * (11 - len(blue))
    if len(red) < 11:
        red = red + [ball] * (11 - len(red))

    return blue[:11], red[:11], ball


# ============================================================
# 5. 메인 파이프라인
# ============================================================

def main():
    video_path = "minimap.mp4"   # 입력 영상
    model_path = "minimap_marker.pt"

    detector = YOLOMarkerDetector(model_path)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1) YOLO 마커 후보
        centers = detector.detect(frame)

        # 2) 팀 분류
        blue, red = classify_teams(frame, centers)

        # 3) 공 검출
        ball = detect_ball(frame)

        # 4) 겹침 보정
        blue, red, ball = fix_overlap(blue, red, ball)

        # 5) 시각화
        for x, y in blue:
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        for x, y in red:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        if ball:
            cv2.circle(frame, ball, 4, (0, 255, 255), -1)

        cv2.imshow("Minimap Tracker (YOLO)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
