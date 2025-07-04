"""High‑level training orchestration script."""
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from config import (
    UNLABELED_DIR,
    CHECKPOINT_DIR,
    NUM_EPOCHS,
    LR,
    MOMENTUM,
    WEIGHT_DECAY,
)
from data.dataset import CocoUnlabeledDataset  # placeholder dataset
from models.detector import build_detector
from .engine import collate_fn, train_one_epoch

def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    # TODO: Replace CocoUnlabeledDataset with your labelled dataset
    dataset = CocoUnlabeledDataset(UNLABELED_DIR)  # placeholder

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: (list(zip(*x))[0], None),  # images only
    )

    model, _ = build_detector()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, loader, optimizer, epoch)
        torch.save(
            model.state_dict(), CHECKPOINT_DIR / f"detector_epoch{epoch}.pth"
        )
    torch.save(
            model.state_dict(), "Model.pth"
        )

if __name__ == "__main__":
    main()
