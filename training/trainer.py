"""Training and validation loops for temporal YOLO models."""

from typing import Dict

import torch
from tqdm import tqdm


__all__ = ['train_epoch', 'validate']


def train_epoch(model, dataloader, optimizer, device, loss_fn, scaler):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_box_loss = 0
    total_cls_loss = 0
    total_dfl_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_data in pbar:
        # Handle both 2-tuple (old) and 3-tuple (new) formats
        if len(batch_data) == 3:
            imgs, labels, augment = batch_data
        else:
            imgs, labels = batch_data
            augment = False

        imgs = imgs.to(device, non_blocking=True).float().div_(255.0)  # GPU normalization
        labels = labels.to(device, non_blocking=True)

        # GPU augmentation (horizontal flip)
        if augment:
            flip_mask = torch.rand(imgs.shape[0], device=device) > 0.5
            if flip_mask.any():
                # Flip images
                imgs[flip_mask] = torch.flip(imgs[flip_mask], dims=[3])

                # Flip bounding box x-coordinates for flipped images
                # Labels format: [batch_idx, class_id, x_center, y_center, width, height]
                for batch_idx in flip_mask.nonzero(as_tuple=True)[0]:
                    label_mask = labels[:, 0] == batch_idx
                    if label_mask.any():
                        labels[label_mask, 2] = 1.0 - labels[label_mask, 2]  # x_center = 1 - x_center

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            preds = model(imgs)
            batch_dict = {
                "batch_idx": labels[:, 0],
                "cls": labels[:, 1],
                "bboxes": labels[:, 2:6],
            }
            loss, loss_items = loss_fn(preds, batch_dict)
            loss = loss.sum()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if hasattr(loss_items, "__len__") and len(loss_items) >= 3:
            total_box_loss += loss_items[0].item() if torch.is_tensor(loss_items[0]) else loss_items[0]
            total_cls_loss += loss_items[1].item() if torch.is_tensor(loss_items[1]) else loss_items[1]
            total_dfl_loss += loss_items[2].item() if torch.is_tensor(loss_items[2]) else loss_items[2]

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "box_loss": total_box_loss / n,
        "cls_loss": total_cls_loss / n,
        "dfl_loss": total_dfl_loss / n,
    }


@torch.no_grad()
def validate(model, dataloader, device, loss_fn):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0

    for batch_data in tqdm(dataloader, desc="Validating"):
        # Handle both 2-tuple (old) and 3-tuple (new) formats
        if len(batch_data) == 3:
            imgs, labels, _ = batch_data  # Ignore augment flag in validation
        else:
            imgs, labels = batch_data

        imgs = imgs.to(device, non_blocking=True).float().div_(255.0)  # GPU normalization
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            preds = model(imgs)
            batch_dict = {
                "batch_idx": labels[:, 0],
                "cls": labels[:, 1],
                "bboxes": labels[:, 2:6],
            }
            loss, _ = loss_fn(preds, batch_dict)
            loss = loss.sum()

        total_loss += loss.item()

    return total_loss / len(dataloader)
