"""Detection and tracking using YOLOv8n + DeepSORT."""

from typing import List, Tuple

import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class PlayerTracker:
    def __init__(self, device: str = "cpu") -> None:
        self.model = YOLO("yolov8n.pt")
        self.model.to(device)
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
        )

    def _foot_position(
        self,
        box: Tuple[float, float, float, float],
    ) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return (0.5 * (x1 + x2), y2)

    def process_frame(self, frame: np.ndarray) -> List[dict]:
        results = self.model.predict(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:  # keep only 'person'
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            detections.append(
                ([x1, y1, x2 - x1, y2 - y1], conf, "person")
            )

        tracks = self.tracker.update_tracks(detections, frame=frame)

        outputs: List[dict] = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = track.to_ltrb()
            foot_x, foot_y = self._foot_position((x1, y1, x2, y2))

            outputs.append(
                {
                    "track_id": track.track_id,
                    "bbox": (x1, y1, x2, y2),
                    "foot": (foot_x, foot_y),
                }
            )

        return outputs
