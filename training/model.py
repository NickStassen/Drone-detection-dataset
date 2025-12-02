"""Model utilities for temporal YOLO training."""

import subprocess
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


__all__ = [
    'MODEL_URLS',
    'download_model',
    'get_first_conv',
    'expand_first_conv',
    'freeze_backbone',
    'get_loss_fn_for_model'
]


MODEL_URLS = {
    "yolov8n": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "yolov8s": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
    "yolov10n": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
    "yolov10s": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
    "yolo11n": None,
    "yolo11s": None,
}


def download_model(model_name: str, weights_dir: Path) -> Path:
    """Download YOLO model weights if not already present."""
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


def get_first_conv(model) -> Tuple[nn.Module, str]:
    """Find the first Conv2d layer with 3 input channels."""
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
    """Expand first conv layer to accept multiple stacked frames."""
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
    """Freeze or unfreeze backbone layers for transfer learning."""
    backbone_end = 10
    for i, layer in enumerate(model.model):
        if i < backbone_end:
            for param in layer.parameters():
                param.requires_grad = not freeze
    print(f"Backbone layers (0-{backbone_end-1}) {'frozen' if freeze else 'unfrozen'}")


def get_loss_fn_for_model(model, model_name: str):
    """
    Select appropriate loss function based on model architecture.

    YOLOv10 uses E2EDetectLoss (handles dict-format predictions).
    YOLOv8/v11 use v8DetectionLoss (handles list-format predictions).
    """
    from ultralytics.utils.loss import v8DetectionLoss

    # Try to import E2EDetectLoss (YOLOv10 loss)
    try:
        from ultralytics.utils.loss import E2EDetectLoss
        has_e2e = True
    except ImportError:
        has_e2e = False

    # Primary detection: check model name
    is_yolov10 = 'yolov10' in model_name.lower()

    # Fallback: check architecture for YOLOv10 indicators
    if not is_yolov10:
        try:
            if hasattr(model.model, 'head'):
                head = model.model.head
                is_yolov10 = hasattr(head, 'aux_pred') or hasattr(head, 'one2one')
        except (AttributeError, TypeError):
            pass

    # Select and return loss function
    if is_yolov10:
        if not has_e2e:
            print("Warning: YOLOv10 detected but E2EDetectLoss unavailable.")
            print("Falling back to v8DetectionLoss (may cause errors).")
            print("Please upgrade ultralytics: pip install ultralytics --upgrade")
            return v8DetectionLoss(model.model)
        print(f"Using E2EDetectLoss for {model_name}")
        return E2EDetectLoss(model.model)
    else:
        print(f"Using v8DetectionLoss for {model_name}")
        return v8DetectionLoss(model.model)
