"""Box drawing helpers."""
from torchvision.utils import draw_bounding_boxes, save_image

def save_detections(img, boxes, labels, scores, categories, out_path, digits=2):
    if len(boxes) == 0:
        return
    lbls = [f"{categories[l]} {s:.{digits}f}" for l, s in zip(labels, scores)]
    drawn = draw_bounding_boxes(
        (img * 255).byte(), boxes=boxes, labels=lbls, width=3
    )
    save_image(drawn, str(out_path))
