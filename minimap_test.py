import cv2
from convert_minimap import extract_minimap_entities


VIDEO_PATH = "minimap.mp4"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        entities = extract_minimap_entities(frame)

        vis = frame.copy()

        bx, by = entities["ball"]
        cv2.circle(vis, (bx, by), 4, (0, 255, 255), -1)

        for x, y in entities["red"]:
            cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

        for x, y in entities["blue"]:
            cv2.circle(vis, (x, y), 4, (255, 0, 0), -1)

        cv2.imshow("convert_minimap test", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
