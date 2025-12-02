#!/usr/bin/env python3
"""
Temporal YOLO Training Script - Optimized with Smart Caching
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path
from typing import Optional
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
import psutil


MODEL_URLS = {
    "yolov8n": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "yolov8s": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
    "yolov10n": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
    "yolov10s": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
    "yolo11n": None,
    "yolo11s": None,
}


class LRUImageCache:
    """Memory-limited LRU cache for images."""

    def __init__(self, max_memory_gb: float = 8.0):
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_bytes = int(max_memory_gb * 1024**3)
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, img: np.ndarray) -> None:
        img_bytes = img.nbytes

        # Evict old entries if needed
        while self.current_bytes + img_bytes > self.max_bytes and self.cache:
            _, old_img = self.cache.popitem(last=False)
            self.current_bytes -= old_img.nbytes

        # Only cache if image fits
        if img_bytes <= self.max_bytes:
            self.cache[key] = img
            self.current_bytes += img_bytes

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __len__(self):
        return len(self.cache)


class TemporalDataset(Dataset):
    """Optimized dataset with smart LRU caching."""

    def __init__(self, images_dir: Path, labels_dir: Path, num_frames: int = 5, img_size: int = 640, augment: bool = True, cache_gb: float = 0):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.num_frames = num_frames
        self.img_size = img_size
        self.augment = augment

        # Smart cache: use LRU with memory limit
        self.cache = LRUImageCache(max_memory_gb=cache_gb) if cache_gb > 0 else None

        # Index by video
        self.video_frames: dict[str, list[int]] = {}
        for f in sorted(self.images_dir.glob("*.jpg")):
            match = re.match(r"(.+)_(\d{5})\.jpg$", f.name)
            if match:
                video, idx = match.group(1), int(match.group(2))
                if video not in self.video_frames:
                    self.video_frames[video] = []
                self.video_frames[video].append(idx)

        # Sort frames and build position lookup
        for video in self.video_frames:
            self.video_frames[video].sort()

        # Create valid samples with precomputed frame sequences
        self.samples: list[tuple[str, int, list[int]]] = []
        for video, frames in self.video_frames.items():
            for i, frame in enumerate(frames):
                if i >= num_frames - 1:
                    frame_seq = [frames[i - (num_frames - 1 - j)] for j in range(num_frames)]
                    self.samples.append((video, frame, frame_seq))

        # Preload all labels into memory (~small, always fits)
        self.labels_cache: dict[str, np.ndarray] = {}
        self._preload_labels()

        cache_status = f", cache: {cache_gb:.1f}GB LRU" if cache_gb > 0 else ""
        print(f"Temporal dataset: {len(self.video_frames)} videos, {len(self.samples)} samples{cache_status}")

    def _preload_labels(self):
        for video, target, _ in self.samples:
            key = f"{video}_{target:05d}"
            if key in self.labels_cache:
                continue
            label_path = self.labels_dir / f"{key}.txt"
            if label_path.exists():
                labels = []
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            labels.append([float(x) for x in parts[:5]])
                self.labels_cache[key] = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
            else:
                self.labels_cache[key] = np.zeros((0, 5), dtype=np.float32)

    def _load_image(self, key: str, path: str) -> np.ndarray:
        # Include size in cache key
        cache_key = f"{key}_{self.img_size}"

        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.cache:
            self.cache.put(cache_key, img)

        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video, target, frame_seq = self.samples[idx]

        # Pre-allocate output array
        stacked = np.empty((self.img_size, self.img_size, 3 * self.num_frames), dtype=np.uint8)

        for i, fidx in enumerate(frame_seq):
            path = str(self.images_dir / f"{video}_{fidx:05d}.jpg")
            img = cv2.imread(path, cv2.IMREAD_COLOR)

            # Only resize if needed (preprocessed images skip this)
            h, w = img.shape[:2]
            if h != self.img_size or w != self.img_size:
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

            # BGR->RGB via slice (no copy)
            stacked[:, :, i * 3 : (i + 1) * 3] = img[:, :, ::-1]

        if self.augment and np.random.rand() > 0.5:
            stacked = np.ascontiguousarray(stacked[:, ::-1, :])

        img_tensor = torch.from_numpy(stacked).permute(2, 0, 1).float().div_(255.0)

        return img_tensor, torch.from_numpy(self.labels_cache[f"{video}_{target:05d}"])

    def print_cache_stats(self):
        if self.cache:
            print(f"  Cache: {len(self.cache)} images, {self.cache.current_bytes/1024**3:.2f}GB, " f"hit rate: {self.cache.hit_rate*100:.1f}%")


def collate_fn(batch):
    imgs, labels_list = zip(*batch)
    imgs = torch.stack(imgs)

    batch_labels = []
    for i, labels in enumerate(labels_list):
        if labels.shape[0] > 0:
            batch_idx = torch.full((labels.shape[0], 1), i, dtype=torch.float32)
            batch_labels.append(torch.cat([batch_idx, labels], dim=1))

    if batch_labels:
        labels = torch.cat(batch_labels, dim=0)
    else:
        labels = torch.zeros((0, 6), dtype=torch.float32)

    return imgs, labels


def download_model(model_name: str, weights_dir: Path) -> Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f"{model_name}.pt"

    if weights_path.exists():
        print(f"Using existing weights: {weights_path}")
        return weights_path

    url = MODEL_URLS.get(model_name)
    if url is None:
        print(f"{model_name} will auto-download via ultralytics")
        return Path(f"{model_name}.pt")

    print(f"Downloading {model_name} from {url}...")
    subprocess.run(["wget", "-q", "-O", str(weights_path), url], check=True)
    print(f"Downloaded to {weights_path}")
    return weights_path


def get_first_conv(model) -> tuple[nn.Module, str]:
    try:
        first_block = model.model.model[0]
        if hasattr(first_block, "conv"):
            return first_block.conv, "model.model.model[0].conv"
    except (AttributeError, IndexError):
        pass

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            return module, name

    raise RuntimeError("Could not find first Conv2d layer with 3 input channels")


def expand_first_conv(model, num_frames: int, init_method: str = "tile") -> None:
    first_conv, conv_path = get_first_conv(model)

    old_weight = first_conv.weight.data
    out_ch, in_ch, kh, kw = old_weight.shape

    assert in_ch == 3, f"Expected 3 input channels, got {in_ch}"

    new_in_ch = 3 * num_frames

    if init_method == "tile":
        new_weight = old_weight.repeat(1, num_frames, 1, 1) / num_frames
    elif init_method == "current":
        new_weight = torch.zeros(out_ch, new_in_ch, kh, kw, device=old_weight.device)
        new_weight[:, :3, :, :] = old_weight
        new_weight[:, 3:, :, :] = torch.randn(out_ch, new_in_ch - 3, kh, kw) * 0.01
    else:
        new_weight = old_weight.repeat(1, num_frames, 1, 1) / num_frames

    new_conv = nn.Conv2d(new_in_ch, out_ch, (kh, kw), stride=first_conv.stride, padding=first_conv.padding, bias=first_conv.bias is not None)
    new_conv.weight.data = new_weight.to(new_conv.weight.device)

    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data.clone()

    parts = conv_path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    setattr(parent, parts[-1], new_conv)

    print(f"Expanded first conv: {in_ch} -> {new_in_ch} channels ({num_frames} frames)")


def freeze_backbone(model, freeze: bool = True) -> None:
    backbone_end = 10
    for i, layer in enumerate(model.model):
        if i < backbone_end:
            for param in layer.parameters():
                param.requires_grad = not freeze
    print(f"Backbone layers (0-{backbone_end-1}) {'frozen' if freeze else 'unfrozen'}")


def train_epoch(model, dataloader, optimizer, device, loss_fn, scaler):
    model.train()
    total_loss = 0
    total_box_loss = 0
    total_cls_loss = 0
    total_dfl_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
    model.eval()
    total_loss = 0

    for imgs, labels in tqdm(dataloader, desc="Validating"):
        imgs = imgs.to(device, non_blocking=True)
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


def get_safe_cache_size() -> float:
    """Determine safe cache size based on available memory."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / 1024**3
    # Use 50% of available, max 12GB, min 2GB
    cache_gb = min(max(available_gb * 0.5, 2.0), 12.0)
    return cache_gb


def train_temporal_yolo(
    model_name: str,
    num_frames: int,
    data_yaml: Path,
    output_dir: Path,
    epochs_phase1: int = 10,
    epochs_phase2: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0",
    init_method: str = "tile",
    weights_dir: Optional[Path] = None,
    phase2_only: bool = False,
    num_workers: int = 4,
    cache_gb: float = -1,  # -1 = auto
) -> None:
    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    weights_dir = weights_dir or Path("./weights")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{device}" if device.isdigit() else device

    # Auto-detect cache size
    if cache_gb < 0:
        cache_gb = get_safe_cache_size()
        print(f"Auto-detected cache size: {cache_gb:.1f}GB")

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    data_root = Path(data_cfg.get("path", data_yaml.parent))
    train_images = data_root / data_cfg["train"]
    val_images = data_root / data_cfg["val"]
    train_labels = train_images.parent.parent / "labels" / train_images.name
    val_labels = val_images.parent.parent / "labels" / val_images.name

    print(f"Train images: {train_images}")
    print(f"Train labels: {train_labels}")
    print(f"Val images: {val_images}")
    print(f"Val labels: {val_labels}")

    # Create datasets with smart caching
    train_dataset = TemporalDataset(train_images, train_labels, num_frames, imgsz, augment=True, cache_gb=cache_gb)
    val_dataset = TemporalDataset(val_images, val_labels, num_frames, imgsz, augment=False, cache_gb=cache_gb * 0.2)  # Smaller val cache

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=False,  # Don't accumulate memory
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    phase1_best = output_dir / f"{model_name}_temporal{num_frames}_phase1" / "best.pt"

    if phase2_only and phase1_best.exists():
        print(f"Loading phase 1 weights: {phase1_best}")
        model = YOLO(str(phase1_best))
    else:
        weights_path = download_model(model_name, weights_dir)
        model = YOLO(str(weights_path))
        expand_first_conv(model.model, num_frames, init_method)

    model.model = model.model.to(device)

    from types import SimpleNamespace

    model.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, nc=1)
    loss_fn = v8DetectionLoss(model.model)

    scaler = torch.amp.GradScaler("cuda")

    # ==================== PHASE 1 ====================
    if not phase2_only:
        print(f"\n{'='*60}")
        print(f"Phase 1: Training head + first conv ({epochs_phase1} epochs)")
        print(f"{'='*60}")

        freeze_backbone(model.model, freeze=True)

        first_conv = model.model.model[0].conv
        for param in first_conv.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.model.parameters()), lr=0.01, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_phase1)

        phase1_dir = output_dir / f"{model_name}_temporal{num_frames}_phase1"
        phase1_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")

        for epoch in range(epochs_phase1):
            print(f"\nEpoch {epoch+1}/{epochs_phase1}")

            train_metrics = train_epoch(model.model, train_loader, optimizer, device, loss_fn, scaler)
            val_loss = validate(model.model, val_loader, device, loss_fn)

            scheduler.step()

            print(f"  Train loss: {train_metrics['loss']:.4f} (box: {train_metrics['box_loss']:.4f}, " f"cls: {train_metrics['cls_loss']:.4f}, dfl: {train_metrics['dfl_loss']:.4f})")
            print(f"  Val loss: {val_loss:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            train_dataset.print_cache_stats()

            torch.save(model.model.state_dict(), phase1_dir / "last.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.model.state_dict(), phase1_dir / "best.pt")
                print(f"  Saved best model (val_loss: {val_loss:.4f})")

        model.model.load_state_dict(torch.load(phase1_dir / "best.pt", weights_only=True))

    # ==================== PHASE 2 ====================
    print(f"\n{'='*60}")
    print(f"Phase 2: Full finetune ({epochs_phase2} epochs)")
    print(f"{'='*60}")

    freeze_backbone(model.model, freeze=False)

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_phase2)

    phase2_dir = output_dir / f"{model_name}_temporal{num_frames}_phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(epochs_phase2):
        print(f"\nEpoch {epoch+1}/{epochs_phase2}")

        train_metrics = train_epoch(model.model, train_loader, optimizer, device, loss_fn, scaler)
        val_loss = validate(model.model, val_loader, device, loss_fn)

        scheduler.step()

        print(f"  Train loss: {train_metrics['loss']:.4f} (box: {train_metrics['box_loss']:.4f}, " f"cls: {train_metrics['cls_loss']:.4f}, dfl: {train_metrics['dfl_loss']:.4f})")
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        train_dataset.print_cache_stats()

        torch.save(model.model.state_dict(), phase2_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.model.state_dict(), phase2_dir / "best.pt")
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

    final_path = output_dir / f"{model_name}_temporal{num_frames}_final.pt"
    torch.save(model.model.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Train temporal YOLO models with frame stacking")
    parser.add_argument("--model", "-m", type=str, required=True, choices=["yolov8n", "yolov8s", "yolov10n", "yolov10s", "yolo11n", "yolo11s"])
    parser.add_argument("--frames", "-f", type=int, default=5)
    parser.add_argument("--data", "-d", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=Path("./runs/temporal"))
    parser.add_argument("--epochs1", type=int, default=10)
    parser.add_argument("--epochs2", type=int, default=50)
    parser.add_argument("--batch", "-b", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--init", type=str, default="tile", choices=["tile", "current", "average"])
    parser.add_argument("--weights-dir", type=Path, default=Path("./weights"))
    parser.add_argument("--phase2-only", action="store_true", help="Skip phase 1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-gb", type=float, default=-1, help="Cache size in GB (-1 = auto)")

    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: Dataset config not found: {args.data}")
        sys.exit(1)

    train_temporal_yolo(
        model_name=args.model,
        num_frames=args.frames,
        data_yaml=args.data,
        output_dir=args.output,
        epochs_phase1=args.epochs1,
        epochs_phase2=args.epochs2,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        init_method=args.init,
        weights_dir=args.weights_dir,
        phase2_only=args.phase2_only,
        num_workers=args.workers,
        cache_gb=args.cache_gb,
    )


if __name__ == "__main__":
    main()
