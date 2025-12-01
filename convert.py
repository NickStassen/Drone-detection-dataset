import scipy.io as sio
import numpy as np
from pathlib import Path
import cv2


def extract_bboxes_from_mat(mat_path: Path) -> list[tuple[float, float, float, float]]:
    """Extract bounding boxes from Halmstad .mat file."""
    mat = sio.loadmat(str(mat_path))
    ws = mat.get("__function_workspace__", np.array([]))
    raw = ws.tobytes()

    bboxes = []
    for i in range(0, len(raw) - 32, 8):
        vals = np.frombuffer(raw[i : i + 32], dtype=np.float64)
        x, y, w, h = vals[0], vals[1], vals[2], vals[3]
        # Relaxed constraints for helicopters (larger objects)
        if 5 < x < 620 and 5 < y < 500 and 10 < w < 300 and 8 < h < 250 and 0.3 < w / h < 3.5:
            bboxes.append((x, y, w, h))

    return bboxes


def convert_dataset(data_dir: Path, output_dir: Path):
    """Convert all videos and labels to YOLO format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)

    video_dir = data_dir / "Video_V"
    stats = {"total_frames": 0, "annotated_frames": 0, "skipped_videos": 0}

    for mat_file in sorted(video_dir.glob("*_LABELS.mat")):
        video_name = mat_file.stem.replace("_LABELS", "") + ".mp4"
        video_path = video_dir / video_name

        if not video_path.exists():
            print(f"Skipping {mat_file.name} - video not found")
            stats["skipped_videos"] += 1
            continue

        bboxes = extract_bboxes_from_mat(mat_file)

        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        n_bboxes = len(bboxes)

        if n_bboxes == 0:
            print(f"Skipping {video_name} - no bboxes extracted")
            cap.release()
            stats["skipped_videos"] += 1
            continue

        print(f"Processing {video_name}: {n_bboxes} bboxes / {frame_count} frames...", end=" ")

        # If fewer bboxes than frames, annotations are sparse
        # We'll assign bboxes to the LAST n frames (object typically enters frame and stays)
        # Or we can assign to FIRST n frames - let's check which makes more sense

        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            frame_name = f"{video_path.stem}_{frame_idx:05d}"
            cv2.imwrite(str(output_dir / "images" / f"{frame_name}.jpg"), frame)

            # Assign bbox if we have one for this frame
            # Assuming bboxes correspond to consecutive frames where object is visible
            if frame_idx < n_bboxes:
                x, y, w, h = bboxes[frame_idx]
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                nw = w / width
                nh = h / height

                with open(output_dir / "labels" / f"{frame_name}.txt", "w") as f:
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                stats["annotated_frames"] += 1
            else:
                # Empty label file - no object in frame
                (output_dir / "labels" / f"{frame_name}.txt").touch()

            stats["total_frames"] += 1

        cap.release()
        print("OK")

    print(f"\nDone! {stats['total_frames']} frames, {stats['annotated_frames']} annotated, {stats['skipped_videos']} videos skipped")


if __name__ == "__main__":
    import shutil

    out = Path("yolo_dataset")
    if out.exists():
        shutil.rmtree(out)
    convert_dataset(Path("Data"), out)
