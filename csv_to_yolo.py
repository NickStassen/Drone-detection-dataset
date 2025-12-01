#!/usr/bin/env python3
"""Convert Halmstad CSV annotations + videos to YOLO format."""

import csv
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import argparse

# Class mapping - all to single class 0 for airspace objects
CLASS_MAP = {"AIRPLANE": 0, "BIRD": 0, "DRONE": 0, "HELICOPTER": 0}


def process_video(
    csv_path: Path,
    video_dir: Path,
    output_dir: Path,
    img_width: int = 640,
    img_height: int = 512,
) -> tuple[int, int]:
    """Extract frames and create YOLO labels for one video."""

    # Parse video name from CSV: V_AIRPLANE_038_LABELS.csv -> V_AIRPLANE_038.mp4
    stem = csv_path.stem.replace("_LABELS", "")
    video_path = video_dir / f"{stem}.mp4"

    if not video_path.exists():
        video_path = video_dir / f"{stem}.avi"
    if not video_path.exists():
        print(f"Video not found: {stem}")
        return 0, 0

    # Load annotations grouped by frame
    annotations: dict[int, list[tuple[str, float, float, float, float]]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            label = row["label"]
            x, y, w, h = float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])
            if frame not in annotations:
                annotations[frame] = []
            annotations[frame].append((label, x, y, w, h))

    if not annotations:
        return 0, 0

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return 0, 0

    # Get actual resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    frames_written = 0
    bboxes_written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in annotations:
            # Save image
            img_name = f"{stem}_{frame_idx:05d}.jpg"
            cv2.imwrite(str(images_out / img_name), frame)

            # Write YOLO label
            label_name = f"{stem}_{frame_idx:05d}.txt"
            with open(labels_out / label_name, "w") as lf:
                for label, x, y, w, h in annotations[frame_idx]:
                    cls_id = CLASS_MAP.get(label, 0)
                    # Convert to YOLO format: normalized center x, center y, width, height
                    cx = (x + w / 2) / actual_w
                    cy = (y + h / 2) / actual_h
                    nw = w / actual_w
                    nh = h / actual_h
                    # Clamp to valid range
                    cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
                    nw, nh = max(0, min(1, nw)), max(0, min(1, nh))
                    lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    bboxes_written += 1

            frames_written += 1

        frame_idx += 1

    cap.release()
    return frames_written, bboxes_written


def main():
    parser = argparse.ArgumentParser(description="Convert Halmstad annotations to YOLO format")
    parser.add_argument("csv_dir", type=Path, help="Directory with CSV label files")
    parser.add_argument("video_dir", type=Path, help="Directory with video files")
    parser.add_argument("output_dir", type=Path, help="Output directory for YOLO dataset")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    csv_files = list(args.csv_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_bboxes = 0

    # Process in parallel
    def worker(csv_path: Path) -> tuple[int, int]:
        return process_video(csv_path, args.video_dir, args.output_dir)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(worker, csv_files))

    for frames, bboxes in results:
        total_frames += frames
        total_bboxes += bboxes

    print(f"\nDone: {total_frames} frames, {total_bboxes} bboxes")
    print(f"Output: {args.output_dir}")

    # Write dataset.yaml for YOLO training
    yaml_content = f"""path: {args.output_dir.absolute()}
train: images
val: images

names:
  0: airspace_object
"""
    (args.output_dir / "dataset.yaml").write_text(yaml_content)
    print(f"Created: {args.output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
