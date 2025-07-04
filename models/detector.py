"""Model builder/factory."""
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from config import DEVICE

def build_detector():
    """Return pretrained FasterRCNN model & category names."""
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights).to(DEVICE)
    model.eval()
    categories = weights.meta["categories"]
    return model, categories
