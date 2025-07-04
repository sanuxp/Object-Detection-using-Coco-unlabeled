"""Run object detection on COCO‑unlabeled images."""
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from config import UNLABELED_DIR, OUTPUT_DIR, CONF_THRESH, DEVICE
from data.dataset import CocoUnlabeledDataset
from models.detector import build_detector
from utils.visualization import save_detections

@torch.inference_mode()
def run_inference():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = CocoUnlabeledDataset(UNLABELED_DIR)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model, categories = build_detector()

    for i, (img, name) in enumerate(dl, 1):
        t0 = time.time()
        preds = model([img[0].to(DEVICE)])[0]
        keep = preds["scores"] >= CONF_THRESH
        filtered = {k: v[keep].cpu() for k, v in preds.items()}

        save_detections(
            img[0],
            filtered["boxes"],
            filtered["labels"],
            filtered["scores"],
            categories,
            OUTPUT_DIR / name[0],
        )

        print(f"[{i}/{len(ds)}] {name[0]}  {len(filtered['boxes'])} objs  "
              f"{time.time() - t0:.2f}s")

if __name__ == "__main__":
    run_inference()
