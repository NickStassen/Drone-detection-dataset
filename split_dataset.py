#!/usr/bin/env python3
"""
Split YOLO dataset into train/val/test by video (prevents temporal leakage).

Usage:
    python split_dataset.py --input yolo_dataset --train 0.8 --val 0.1 --test 0.1
"""

import argparse
import re
import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def get_video_from_filename(filename: str) -> str | None:
    """Extract video name from filename like 'V_AIRPLANE_038_00015.jpg'"""
    match = re.match(r"(.+)_(\d{5})\.(jpg|txt)$", filename)
    if match:
        return match.group(1)
    return None


def split_dataset(
    input_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    input_dir = Path(input_dir)
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        print(f"Warning: Ratios sum to {total}, normalizing...")
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

    # Check if already split
    if (images_dir / "train").exists():
        print("Dataset appears to already be split (train folder exists).")
        response = input("Flatten and re-split? (y/n): ")
        if response.lower() != "y":
            print("Aborted.")
            return

        # Flatten existing split
        print("Flattening existing split...")
        for split in ["train", "val", "test"]:
            split_img = images_dir / split
            split_lbl = labels_dir / split
            if split_img.exists():
                for f in split_img.glob("*.*"):
                    shutil.move(f, images_dir / f.name)
                split_img.rmdir()
            if split_lbl.exists():
                for f in split_lbl.glob("*.*"):
                    shutil.move(f, labels_dir / f.name)
                split_lbl.rmdir()

    # Group images by video
    video_frames: dict[str, list[Path]] = defaultdict(list)

    for img_path in images_dir.glob("*.jpg"):
        video = get_video_from_filename(img_path.name)
        if video:
            video_frames[video].append(img_path)

    if not video_frames:
        # Try png
        for img_path in images_dir.glob("*.png"):
            video = get_video_from_filename(img_path.name)
            if video:
                video_frames[video].append(img_path)

    if not video_frames:
        print("Error: No images found matching pattern *_XXXXX.jpg")
        return

    videos = list(video_frames.keys())
    total_videos = len(videos)
    total_frames = sum(len(frames) for frames in video_frames.values())

    print(f"\nDataset statistics:")
    print(f"  Total videos: {total_videos}")
    print(f"  Total frames: {total_frames}")
    print(f"  Avg frames/video: {total_frames / total_videos:.1f}")

    # Shuffle and split videos
    random.seed(seed)
    random.shuffle(videos)

    n_train = int(total_videos * train_ratio)
    n_val = int(total_videos * val_ratio)

    train_videos = set(videos[:n_train])
    val_videos = set(videos[n_train : n_train + n_val])
    test_videos = set(videos[n_train + n_val :])

    # Create directories
    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(exist_ok=True)
        (labels_dir / split).mkdir(exist_ok=True)

    # Move files
    splits = {
        "train": train_videos,
        "val": val_videos,
        "test": test_videos,
    }

    counts = {"train": 0, "val": 0, "test": 0}

    print("\nMoving files...")
    for video, frames in tqdm(video_frames.items()):
        # Determine split
        if video in train_videos:
            split = "train"
        elif video in val_videos:
            split = "val"
        else:
            split = "test"

        for img_path in frames:
            # Move image
            dst_img = images_dir / split / img_path.name
            shutil.move(img_path, dst_img)

            # Move label if exists
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                dst_lbl = labels_dir / split / label_path.name
                shutil.move(label_path, dst_lbl)

            counts[split] += 1

    # Print summary
    print(f"\nSplit complete:")
    print(f"  Train: {len(train_videos)} videos, {counts['train']} frames ({counts['train']/total_frames*100:.1f}%)")
    print(f"  Val:   {len(val_videos)} videos, {counts['val']} frames ({counts['val']/total_frames*100:.1f}%)")
    print(f"  Test:  {len(test_videos)} videos, {counts['test']} frames ({counts['test']/total_frames*100:.1f}%)")

    # Update/create dataset.yaml
    yaml_content = f"""path: {input_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: threat
"""
    yaml_path = input_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nUpdated {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset into train/val/test by video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 80/10/10 split
  python split_dataset.py --input yolo_dataset
  
  # Custom split
  python split_dataset.py --input yolo_dataset --train 0.7 --val 0.15 --test 0.15
  
  # Reproducible with specific seed
  python split_dataset.py --input yolo_dataset --seed 123
""",
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--val", type=float, default=0.1, help="Validation ratio (default: 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}")
        return

    split_dataset(
        input_dir=args.input,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
