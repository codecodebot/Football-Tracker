"""End-to-end pipeline for player mapping evaluation."""

from typing import List

import cv2
import numpy as np
import torch
from google.colab.patches import cv2_imshow

import project.config as config
from project.detect_track import PlayerTracker
from project.evaluate import evaluate_frame, summarize_metrics
from project.matcher import hungarian_match
from project.minimap_gt import extract_minimap_points
from project.projection import (
    compute_homography,
    create_field_map,
    project_points,
)
from project.train_mapping import load_model, train_model


def _log_training_data(rows: List[List[float]], path: str) -> None:
    if not rows:
        return
    data = np.array(rows, dtype=np.float32)
    np.savez(path, inputs=data[:, :2], targets=data[:, 2:])


def run_pipeline() -> None:
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {config.VIDEO_PATH}")

    tracker = PlayerTracker(device="cpu")

    x, y, w, h = config.MINIMAP_ROI
    homography = compute_homography(
        (w, h),
        (config.FIELD_MAP_WIDTH, config.FIELD_MAP_HEIGHT),
    )
    field_map = create_field_map(
        config.FIELD_MAP_WIDTH,
        config.FIELD_MAP_HEIGHT,
    )

    frame_idx = 0
    training_rows: List[List[float]] = []
    per_frame_metrics = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if config.FRAME_SKIP > 1 and frame_idx % config.FRAME_SKIP != 0:
            continue

        # Prediction from main video
        tracks = tracker.process_frame(frame)
        pred_points = [t["foot"] for t in tracks]
        pred_points = project_points(pred_points, homography)

        # Ground truth from minimap
        gt_points = extract_minimap_points(
            frame,
            config.MINIMAP_ROI,
            config.HSV_LOWER,
            config.HSV_UPPER,
        )
        gt_points = project_points(gt_points, homography)

        pred_arr = np.array(pred_points, dtype=np.float32)
        gt_arr = np.array(gt_points, dtype=np.float32)

        matches, _, _ = hungarian_match(pred_arr, gt_arr)

        frame_metrics = evaluate_frame(pred_arr, gt_arr, matches)
        per_frame_metrics.append(frame_metrics)

        # Collect training data
        for pred_idx, gt_idx in matches:
            x_img, y_img = tracks[pred_idx]["foot"]
            x_field, y_field = gt_arr[gt_idx]
            training_rows.append([x_img, y_img, x_field, y_field])

        # Optional visualization (Colab)
        if config.ENABLE_VISUALS:
            vis = field_map.copy()
            for pt in pred_arr:
                cv2.circle(
                    vis,
                    (int(pt[0]), int(pt[1])),
                    4,
                    (0, 0, 255),
                    -1,
                )
            for pt in gt_arr:
                cv2.circle(
                    vis,
                    (int(pt[0]), int(pt[1])),
                    4,
                    (255, 255, 0),
                    -1,
                )
            cv2_imshow(vis)

    cap.release()

    _log_training_data(training_rows, config.LOG_DATA_PATH)

    overall = summarize_metrics(per_frame_metrics)
    print("Baseline metrics:", overall)

    # Optional learning-based correction
    if len(training_rows) > 10:
        train_model(
            config.LOG_DATA_PATH,
            config.MODEL_PATH,
            config.TRAINING.batch_size,
            config.TRAINING.epochs,
            config.TRAINING.lr,
            config.TRAINING.weight_decay,
            config.TRAINING.val_split,
        )

        model = load_model(config.MODEL_PATH)
        corrected_metrics = _evaluate_with_model(
            model,
            training_rows,
        )
        print("Corrected metrics:", corrected_metrics)


def _evaluate_with_model(model, training_rows):
    data = np.array(training_rows, dtype=np.float32)
    inputs = data[:, :2]
    targets = data[:, 2:]

    preds = model(torch.tensor(inputs)).detach().numpy()
    errors = np.linalg.norm(preds - targets, axis=1)

    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


if __name__ == "__main__":
    run_pipeline()
