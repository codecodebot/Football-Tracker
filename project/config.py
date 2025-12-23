"""Configuration for the eFootball player mapping evaluation project."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2


VIDEO_PATH = "/content/efootball.mp4"

MINIMAP_ROI = (0, 0, 300, 200)

FIELD_MAP_WIDTH = 680
FIELD_MAP_HEIGHT = 1050

HSV_LOWER = (0, 0, 200)
HSV_UPPER = (180, 60, 255)

FRAME_SKIP = 1
ENABLE_VISUALS = False

TRAINING = TrainingConfig()

LOG_DATA_PATH = "/content/project/training_data.npz"
MODEL_PATH = "/content/project/model.pth"
"""Configuration for the eFootball player mapping evaluation project."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2


VIDEO_PATH = "/content/efootball.mp4"

MINIMAP_ROI = (0, 0, 300, 200)

FIELD_MAP_WIDTH = 680
FIELD_MAP_HEIGHT = 1050

HSV_LOWER = (0, 0, 200)
HSV_UPPER = (180, 60, 255)

FRAME_SKIP = 1
ENABLE_VISUALS = False

TRAINING = TrainingConfig()

LOG_DATA_PATH = "/content/project/training_data.npz"
MODEL_PATH = "/content/project/model.pth"
