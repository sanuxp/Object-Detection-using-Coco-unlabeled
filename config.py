"""Global configuration."""
from pathlib import Path
import torch

# ===== Paths =====
PROJ_ROOT = Path(__file__).parent
COCO_ROOT = PROJ_ROOT / "coco"
UNLABELED_DIR = COCO_ROOT / "unlabeled2017"
OUTPUT_DIR = PROJ_ROOT / "detections"
CHECKPOINT_DIR = PROJ_ROOT / "checkpoints"

# ===== Training & Inference =====
CONF_THRESH: float = 0.5
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS: int = 10
LR: float = 5e-3
MOMENTUM: float = 0.9
WEIGHT_DECAY: float = 1e-4
