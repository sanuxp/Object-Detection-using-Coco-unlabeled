"""Reusable training / evaluation loops."""
import torch
from typing import List, Dict
from config import DEVICE

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, loader, optimizer, epoch: int):
    model.train()
    for imgs, targets in loader:
        imgs = [img.to(DEVICE) for img in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}  loss={loss.item():.4f}")

@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    losses = []
    for imgs, targets in loader:
        imgs = [img.to(DEVICE) for img in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        losses.append(sum(loss_dict.values()).item())
    return sum(losses) / len(losses)
