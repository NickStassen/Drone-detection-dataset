#!/usr/bin/env python3
"""Preprocess dataset to target resolution - run ONCE before training."""

import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
import shutil


def process_image(args):
    src, dst, size = args
    try:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        if h != size or w != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        print(f"Error processing {src}: {e}")
        return False


def preprocess_dataset(input_dir: Path, output_dir: Path, img_size: int = 640, workers: int = 16):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output structure
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Collect all tasks
    image_tasks = []
    label_tasks = []

    for split in ["train", "val", "test"]:
        src_img = input_dir / "images" / split
        dst_img = output_dir / "images" / split
        src_lbl = input_dir / "labels" / split
        dst_lbl = output_dir / "labels" / split

        if not src_img.exists():
            continue

        for img_path in src_img.glob("*.jpg"):
            image_tasks.append((img_path, dst_img / img_path.name, img_size))

            lbl_src = src_lbl / f"{img_path.stem}.txt"
            lbl_dst = dst_lbl / f"{img_path.stem}.txt"
            if lbl_src.exists():
                label_tasks.append((lbl_src, lbl_dst))

    print(f"Preprocessing {len(image_tasks)} images to {img_size}×{img_size}...")

    # Process images with thread pool (IO-bound, threads are fine)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(process_image, image_tasks), total=len(image_tasks)))

    print(f"Processed {sum(results)}/{len(image_tasks)} images")

    # Copy labels (fast, just do it serially)
    print(f"Copying {len(label_tasks)} label files...")
    for src, dst in tqdm(label_tasks):
        shutil.copy2(src, dst)

    # Create dataset.yaml
    yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: threat
"""
    (output_dir / "dataset.yaml").write_text(yaml_content)
    print(f"\nDone! Dataset saved to: {output_dir}")
    print(f"Run training with: --data {output_dir}/dataset.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset to fixed resolution")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input dataset directory")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output dataset directory")
    parser.add_argument("--size", "-s", type=int, default=640, help="Target image size (default: 640)")
    parser.add_argument("--workers", "-w", type=int, default=16, help="Number of worker threads")
    args = parser.parse_args()

    preprocess_dataset(args.input, args.output, args.size, args.workers)
