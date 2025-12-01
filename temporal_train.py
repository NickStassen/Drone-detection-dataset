#!/usr/bin/env python3
"""
Temporal YOLO Training Script

Trains YOLOv8n, YOLOv10n, or YOLO11n with temporal frame stacking.
Modifies first conv layer to accept N stacked frames as input.

Usage:
    python train_temporal_yolo.py --model yolov8n --frames 5 --data dataset.yaml
    python train_temporal_yolo.py --model yolov10n --frames 3 --data dataset.yaml
    python train_temporal_yolo.py --model yolo11n --frames 5 --data dataset.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# Model download URLs
MODEL_URLS = {
    "yolov8n": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "yolov8s": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
    "yolov10n": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
    "yolov10s": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
    "yolo11n": None,  # Auto-downloads via ultralytics
    "yolo11s": None,
}


def download_model(model_name: str, weights_dir: Path) -> Path:
    """Download pretrained weights if not present."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f"{model_name}.pt"

    if weights_path.exists():
        print(f"Using existing weights: {weights_path}")
        return weights_path

    url = MODEL_URLS.get(model_name)
    if url is None:
        # YOLO11 auto-downloads via ultralytics
        print(f"{model_name} will auto-download via ultralytics")
        return Path(f"{model_name}.pt")

    print(f"Downloading {model_name} from {url}...")
    subprocess.run(["wget", "-q", "-O", str(weights_path), url], check=True)
    print(f"Downloaded to {weights_path}")
    return weights_path


def get_first_conv(model) -> tuple[nn.Module, str]:
    """Find the first Conv2d layer in the model."""
    # YOLOv8/v10/v11 structure: model.model.model[0].conv
    try:
        first_block = model.model.model[0]
        if hasattr(first_block, "conv"):
            return first_block.conv, "model.model.model[0].conv"
    except (AttributeError, IndexError):
        pass

    # Fallback: search recursively
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            return module, name

    raise RuntimeError("Could not find first Conv2d layer with 3 input channels")


def expand_first_conv(model, num_frames: int, init_method: str = "tile") -> None:
    """
    Expand first conv layer from 3 channels to 3*num_frames channels.

    Args:
        model: YOLO model
        num_frames: Number of temporal frames to stack
        init_method: Weight initialization method
            - "tile": Tile pretrained weights (each frame starts with RGB features)
            - "current": Current frame uses pretrained, others small random
            - "average": Average pretrained weights across new channels
    """
    first_conv, conv_path = get_first_conv(model)

    old_weight = first_conv.weight.data  # [out_ch, 3, kH, kW]
    out_ch, in_ch, kh, kw = old_weight.shape

    assert in_ch == 3, f"Expected 3 input channels, got {in_ch}"

    new_in_ch = 3 * num_frames

    # Initialize new weights
    if init_method == "tile":
        # Tile weights and scale down
        new_weight = old_weight.repeat(1, num_frames, 1, 1) / num_frames
    elif init_method == "current":
        # Current frame (first 3 channels) = pretrained, rest = small random
        new_weight = torch.zeros(out_ch, new_in_ch, kh, kw, device=old_weight.device)
        new_weight[:, :3, :, :] = old_weight
        new_weight[:, 3:, :, :] = torch.randn(out_ch, new_in_ch - 3, kh, kw) * 0.01
    elif init_method == "average":
        # Spread pretrained weights across all frames
        new_weight = old_weight.repeat(1, num_frames, 1, 1) / num_frames
    else:
        raise ValueError(f"Unknown init_method: {init_method}")

    # Create new conv layer
    new_conv = nn.Conv2d(new_in_ch, out_ch, (kh, kw), stride=first_conv.stride, padding=first_conv.padding, bias=first_conv.bias is not None)
    new_conv.weight.data = new_weight.to(new_conv.weight.device)

    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data.clone()

    # Replace in model
    # Navigate to parent and replace
    parts = conv_path.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_conv)

    print(f"Expanded first conv: {in_ch} -> {new_in_ch} channels ({num_frames} frames)")
    print(f"  Weight shape: {old_weight.shape} -> {new_weight.shape}")
    print(f"  Init method: {init_method}")


def create_temporal_dataset_yaml(original_yaml: Path, output_yaml: Path, num_frames: int) -> None:
    """
    Create a modified dataset.yaml noting temporal configuration.
    The actual frame stacking happens in the dataloader.
    """
    import yaml

    with open(original_yaml) as f:
        config = yaml.safe_load(f)

    # Add temporal info (for documentation, doesn't affect YOLO directly)
    config["temporal_frames"] = num_frames
    config["notes"] = f"Temporal dataset with {num_frames} stacked frames"

    with open(output_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Created temporal dataset config: {output_yaml}")


def freeze_backbone(model, freeze: bool = True) -> None:
    """Freeze/unfreeze backbone layers (keep head trainable)."""
    # YOLOv8 structure: model[0-9] is backbone, model[10+] is head
    backbone_end = 10  # Approximate

    for i, layer in enumerate(model.model.model):
        if i < backbone_end:
            for param in layer.parameters():
                param.requires_grad = not freeze

    status = "frozen" if freeze else "unfrozen"
    print(f"Backbone layers (0-{backbone_end-1}) {status}")


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
) -> None:
    """
    Two-phase training for temporal YOLO:
    1. Freeze backbone, train first conv + head
    2. Unfreeze all, finetune with lower LR
    """
    from ultralytics import YOLO

    weights_dir = weights_dir or Path("./weights")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download/locate weights
    weights_path = download_model(model_name, weights_dir)

    # Load model
    print(f"\nLoading {model_name}...")
    model = YOLO(str(weights_path))

    # Expand first conv for temporal input
    expand_first_conv(model.model, num_frames, init_method)

    # Save modified model architecture
    modified_weights = output_dir / f"{model_name}_temporal{num_frames}_init.pt"
    torch.save(model.model.state_dict(), modified_weights)
    print(f"Saved initial temporal weights: {modified_weights}")

    # Phase 1: Freeze backbone, train head + first conv
    print(f"\n{'='*60}")
    print(f"Phase 1: Training head + first conv ({epochs_phase1} epochs)")
    print(f"{'='*60}")

    freeze_backbone(model.model, freeze=True)

    # Ensure first conv is trainable
    first_conv, _ = get_first_conv(model.model)
    for param in first_conv.parameters():
        param.requires_grad = True

    results1 = model.train(
        data=str(data_yaml),
        epochs=epochs_phase1,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(output_dir),
        name=f"{model_name}_temporal{num_frames}_phase1",
        exist_ok=True,
        pretrained=False,  # We already loaded weights
        lr0=0.01,
        lrf=0.1,
        warmup_epochs=3,
        close_mosaic=epochs_phase1,  # Disable mosaic for temporal consistency
    )

    # Phase 2: Unfreeze all, finetune
    print(f"\n{'='*60}")
    print(f"Phase 2: Full finetune ({epochs_phase2} epochs)")
    print(f"{'='*60}")

    # Load best weights from phase 1
    phase1_best = output_dir / f"{model_name}_temporal{num_frames}_phase1" / "weights" / "best.pt"
    if phase1_best.exists():
        model = YOLO(str(phase1_best))

    freeze_backbone(model.model, freeze=False)

    results2 = model.train(
        data=str(data_yaml),
        epochs=epochs_phase2,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(output_dir),
        name=f"{model_name}_temporal{num_frames}_phase2",
        exist_ok=True,
        lr0=0.001,  # Lower LR for finetuning
        lrf=0.01,
        warmup_epochs=0,
        close_mosaic=10,
    )

    # Export final model
    final_weights = output_dir / f"{model_name}_temporal{num_frames}_final.pt"
    phase2_best = output_dir / f"{model_name}_temporal{num_frames}_phase2" / "weights" / "best.pt"
    if phase2_best.exists():
        import shutil

        shutil.copy(phase2_best, final_weights)
        print(f"\nFinal model saved: {final_weights}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Train temporal YOLO models with frame stacking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train YOLOv8n with 5 temporal frames
  python train_temporal_yolo.py --model yolov8n --frames 5 --data dataset.yaml

  # Train YOLOv10n with 3 frames, custom batch size
  python train_temporal_yolo.py --model yolov10n --frames 3 --data dataset.yaml --batch 32

  # Quick test with fewer epochs
  python train_temporal_yolo.py --model yolo11n --frames 5 --data dataset.yaml --epochs1 2 --epochs2 5
        """,
    )

    parser.add_argument("--model", "-m", type=str, required=True, choices=["yolov8n", "yolov8s", "yolov10n", "yolov10s", "yolo11n", "yolo11s"], help="Model architecture to use")
    parser.add_argument("--frames", "-f", type=int, default=5, help="Number of temporal frames to stack (default: 5)")
    parser.add_argument("--data", "-d", type=Path, required=True, help="Path to dataset.yaml")
    parser.add_argument("--output", "-o", type=Path, default=Path("./runs/temporal"), help="Output directory (default: ./runs/temporal)")
    parser.add_argument("--epochs1", type=int, default=10, help="Epochs for phase 1 (frozen backbone) (default: 10)")
    parser.add_argument("--epochs2", type=int, default=50, help="Epochs for phase 2 (full finetune) (default: 50)")
    parser.add_argument("--batch", "-b", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device (default: 0)")
    parser.add_argument("--init", type=str, default="tile", choices=["tile", "current", "average"], help="Weight init method for expanded conv (default: tile)")
    parser.add_argument("--weights-dir", type=Path, default=Path("./weights"), help="Directory to store downloaded weights")

    args = parser.parse_args()

    # Validate data yaml exists
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
    )


if __name__ == "__main__":
    main()
