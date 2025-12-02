"""Custom batch sampler for temporal locality."""
import random
from typing import Iterator, List
from torch.utils.data import Sampler


class TemporalBatchSampler(Sampler[List[int]]):
    """Groups temporally adjacent samples to improve cache hits."""

    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Group samples by video
        self.video_groups = {}
        for idx, (video, target, frame_seq) in enumerate(dataset.samples):
            if video not in self.video_groups:
                self.video_groups[video] = []
            self.video_groups[video].append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle videos
        videos = list(self.video_groups.keys())
        if self.shuffle:
            random.shuffle(videos)

        # Yield batches with temporal locality
        batch = []
        for video in videos:
            indices = self.video_groups[video].copy()
            if self.shuffle:
                random.shuffle(indices)

            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        # Handle final partial batch
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
