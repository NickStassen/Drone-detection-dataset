"""
Temporal YOLO Training Package

Provides modular components for training YOLO models with temporal frame stacking.
"""

from .data import LRUImageCache, TemporalDataset, collate_fn
from .model import (
    MODEL_URLS,
    download_model,
    expand_first_conv,
    freeze_backbone,
    get_loss_fn_for_model
)
from .trainer import train_epoch, validate
from .utils import get_safe_cache_size

__all__ = [
    # Data
    'LRUImageCache',
    'TemporalDataset',
    'collate_fn',
    # Model
    'MODEL_URLS',
    'download_model',
    'expand_first_conv',
    'freeze_backbone',
    'get_loss_fn_for_model',
    # Trainer
    'train_epoch',
    'validate',
    # Utils
    'get_safe_cache_size',
]

__version__ = '1.0.0'
