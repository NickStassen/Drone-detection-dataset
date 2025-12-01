# verify_random.py
import cv2
import random
from pathlib import Path

labels_dir = Path("yolo_dataset/labels")
images_dir = Path("yolo_dataset/images")

# Find non-empty label files
annotated = [f for f in labels_dir.glob("*.txt") if f.stat().st_size > 0]
samples = random.sample(annotated, min(5, len(annotated)))

for label_path in samples:
    img_path = images_dir / label_path.name.replace(".txt", ".jpg")
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    with open(label_path) as f:
        parts = f.read().strip().split()
        cx, cy, bw, bh = map(float, parts[1:])

    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    out_name = f"verify_{label_path.stem}.jpg"
    cv2.imwrite(out_name, img)
    print(f"Saved {out_name}")
