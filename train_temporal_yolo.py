#!/usr/bin/env python3
"""
Temporal YOLO Training Script - Optimized with Smart Caching
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader

from training import (
    TemporalDataset,
    TemporalBatchSampler,
    collate_fn,
    download_model,
    expand_first_conv,
    freeze_backbone,
    get_loss_fn_for_model,
    train_epoch,
    validate,
    get_safe_cache_size
)


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

    train_sampler = TemporalBatchSampler(train_dataset, batch_size, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,  # Cache survives across epochs
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

    model.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, nc=1)
    loss_fn = get_loss_fn_for_model(model, model_name)

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

        # Very low LR to prevent gradient explosion with expanded conv
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.model.parameters()), lr=0.0001, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs_phase1)

        # Warmup: gradually increase LR over first 3 epochs
        warmup_epochs = min(3, epochs_phase1)
        warmup_factor = 0.01  # Start at 1% of target LR

        phase1_dir = output_dir / f"{model_name}_temporal{num_frames}_phase1"
        phase1_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")

        for epoch in range(epochs_phase1):
            print(f"\nEpoch {epoch+1}/{epochs_phase1}")

            # Apply warmup to learning rate
            if epoch < warmup_epochs:
                warmup_lr = 0.001 * (warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"  Warmup LR: {warmup_lr:.6f}")

            train_metrics = train_epoch(model.model, train_loader, optimizer, device, loss_fn, scaler)
            val_loss = validate(model.model, val_loader, device, loss_fn)

            if epoch >= warmup_epochs:
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
    parser.add_argument("--workers", type=int, default=4)
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
