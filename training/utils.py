"""Utility functions for temporal YOLO training."""

import psutil


__all__ = ['get_safe_cache_size']


def get_safe_cache_size() -> float:
    """Determine safe cache size based on available memory."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / 1024**3
    # Use 50% of available, max 12GB, min 2GB
    cache_gb = min(max(available_gb * 0.5, 2.0), 12.0)
    return cache_gb
