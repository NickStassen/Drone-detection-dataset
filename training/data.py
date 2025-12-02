"""Dataset and data loading utilities for temporal YOLO training."""

import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


__all__ = ['LRUImageCache', 'TemporalDataset', 'collate_fn']


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
