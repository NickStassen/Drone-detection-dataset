import scipy.io as sio
import numpy as np
from pathlib import Path


def find_frame_bbox_mapping(mat_path: Path):
    mat = sio.loadmat(str(mat_path))
    ws = mat.get("__function_workspace__", np.array([]))
    raw = ws.tobytes()

    print(f"=== Analyzing {mat_path.name} ===")

    # MATLAB cell arrays store sizes before data
    # Look for an array of small integers that could be "bbox count per frame"
    # For BIRD_005: 302 frames, pattern like [1,2,4,2,4,2,4...]

    # Search for sequences that look like bbox counts
    print("Searching for bbox-count-per-frame arrays...")

    for dtype in [np.int32, np.uint32, np.int16, np.uint16, np.uint8]:
        for i in range(0, min(len(raw) - 100, 60000), 4):
            try:
                # Read potential count array
                size = 50  # Check 50 values
                vals = np.frombuffer(raw[i : i + size * np.dtype(dtype).itemsize], dtype=dtype)

                # Look for pattern: all values small (0-20), mix of values, sum reasonable
                if len(vals) >= 20 and all(0 <= v <= 30 for v in vals[:20]) and 1 <= np.mean(vals[:20]) <= 20 and len(set(vals[:20].tolist())) > 2:  # Not all same value

                    # Check if sum could match total bbox count
                    total = sum(vals[:302]) if len(vals) >= 302 else sum(vals)

                    print(f"\n  Offset {i} ({dtype.__name__}):")
                    print(f"    First 30 values: {vals[:30].tolist()}")
                    print(f"    Sum of first 302: {total if len(vals) >= 302 else 'N/A'}")

            except:
                pass

    # Also look for the 336-byte stride pattern location
    # This might tell us where single-bbox-per-frame data is
    print("\n\nLooking for stride markers...")
    for i in range(60000, min(len(raw) - 400, 70000)):
        # Check if this could be start of regular 336-byte bbox array
        bboxes_found = 0
        for j in range(10):
            offset = i + j * 336
            if offset + 32 > len(raw):
                break
            vals = np.frombuffer(raw[offset : offset + 32], dtype=np.float64)
            x, y, w, h = vals[0], vals[1], vals[2], vals[3]
            if 5 < x < 620 and 5 < y < 500 and 10 < w < 300 and 8 < h < 250:
                bboxes_found += 1

        if bboxes_found >= 8:
            print(f"  Regular bbox array likely starts at offset {i}")
            # Show structure before this
            pre = raw[i - 64 : i]
            pre_ints = np.frombuffer(pre, dtype=np.int32)
            print(f"    64 bytes before (int32): {pre_ints}")
            break


# Test on both files
find_frame_bbox_mapping(Path("Data/Video_V/V_AIRPLANE_001_LABELS.mat"))
print("\n" + "=" * 80 + "\n")
find_frame_bbox_mapping(Path("Data/Video_V/V_BIRD_005_LABELS.mat"))
