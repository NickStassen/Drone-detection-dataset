# Temporal YOLO DeepStream Deployment Plan

## Executive Summary

**Goal:** Deploy 5-frame temporal YOLO (15-channel input) in a multi-camera DeepStream GStreamer pipeline with TensorRT acceleration.

**Feasibility:** ✅ YES - Fully feasible with high confidence (95%)

**Timeline:** 7-12 days

**Key Strategy:** RAM-based frame buffering with GStreamer probe callbacks

---

## 1. Architecture Overview

### Current State
- **Temporal YOLO:** 5 stacked RGB frames → 15-channel input `(batch, 15, 640, 640)`
- **Modified architecture:** First Conv2d expanded from 3→15 channels (`training/model.py:67-100`)
- **Frame stacking:** Channel-wise concatenation in `training/data.py:140-149`
- **Training complete:** Two-phase approach with pretrained weight initialization

### Target Deployment
- **Hardware:** dGPU (RTX 3000/4000 series)
- **Pipeline:** Multi-camera real-time (30fps per stream)
- **Inference:** TensorRT engine with FP16 precision
- **Buffering:** CPU RAM (6MB per stream, 300MB for 50 streams)

### Data Flow

```
Camera Stream → uridecodebin → nvvideoconvert → capsfilter (RGBA)
                                                      ↓
                                          [PROBE 1: Frame Capture]
                                          Store in RAM buffer (deque)
                                                      ↓
                                              nvstreammux
                                                      ↓
                                          [PROBE 2: Frame Stacking]
                                          Stack 5 frames → 15 channels
                                                      ↓
                                                  nvinfer
                                            (TensorRT engine)
                                        Input: (batch, 15, 640, 640)
                                        Output: (batch, N, 6)
                                                      ↓
                                           Detection callbacks
```

---

## 2. ONNX Export Implementation

### File to Create: `export_temporal_yolo.py`

**Location:** `/home/nick/Workspace/Drone-detection-dataset/export_temporal_yolo.py`

**Based on:** `/home/nick/Workspace/cluster-server-gstreamer/DeepStream-Yolo2/utils/export_yoloV8.py`

### Key Modifications

1. **Change input shape** from `(batch, 3, H, W)` to `(batch, 15, H, W)`:
   ```python
   # Line 94 in original → modify to:
   onnx_input_im = torch.zeros(args.batch, 15, *img_size).to(device)
   ```

2. **Add validation** that model has 15-channel first conv:
   ```python
   from training.model import get_first_conv

   first_conv, conv_path = get_first_conv(model)
   assert first_conv.in_channels == 15, \
       f"Expected 15 input channels, got {first_conv.in_channels}"
   ```

3. **Reuse DeepStreamOutput wrapper** (no changes needed - already compatible)

4. **Load trained weights** instead of pretrained:
   ```python
   # args.weights should point to your Phase 2 trained checkpoint
   # Example: runs/temporal/phase2_final.pt
   ```

### Export Command

```bash
poetry run python export_temporal_yolo.py \
  --weights runs/temporal/phase2_final.pt \
  --size 640 \
  --opset 17 \
  --dynamic \
  --simplify
```

**Output:** `yolov8n_temporal5.onnx` with input shape `(batch, 15, 640, 640)`

### Validation Script: `test_onnx_export.py`

```python
import torch
import onnxruntime as ort
import numpy as np

# Load PyTorch model
model = torch.load('runs/temporal/phase2_final.pt')
model.eval()

# Load ONNX model
session = ort.InferenceSession('yolov8n_temporal5.onnx')

# Create random 15-channel input
dummy_input = torch.randn(1, 15, 640, 640)

# PyTorch inference
with torch.no_grad():
    torch_out = model(dummy_input)

# ONNX inference
onnx_out = session.run(None, {'input': dummy_input.numpy()})

# Compare outputs
np.testing.assert_allclose(torch_out.numpy(), onnx_out[0], rtol=1e-3, atol=1e-4)
print("✅ ONNX export validated - outputs match PyTorch")
```

---

## 3. TensorRT Engine Generation

### Build Script: `build_trt_engine.sh`

```bash
#!/bin/bash

/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov8n_temporal5.onnx \
  --saveEngine=yolov8n_temporal5_fp16.engine \
  --explicitBatch \
  --minShapes=input:1x15x640x640 \
  --optShapes=input:4x15x640x640 \
  --maxShapes=input:16x15x640x640 \
  --fp16 \
  --workspace=4096 \
  --verbose \
  --dumpLayerInfo \
  --exportLayerInfo=layer_info.json
```

**Key Points:**
- **Dynamic batching:** 1-16 streams flexible
- **FP16 precision:** 2x speedup, minimal accuracy loss
- **TensorRT fully supports 15-channel Conv2d** - just matrix multiplication, no special handling needed

### Performance Expectations

| GPU | Batch Size | Inference Time | Throughput |
|-----|------------|----------------|------------|
| RTX 3060 | 1 | 10-12ms | 80-100 FPS |
| RTX 3080 | 1 | 6-8ms | 125-167 FPS |
| RTX 4090 | 1 | 3-5ms | 200-333 FPS |
| RTX 3080 (batch 4) | 4 | 12-15ms | 67-83 FPS/stream |

**Real-time target:** 33ms per frame @ 30fps → Plenty of margin ✅

---

## 4. Frame Buffering Architecture

### Design Choice: RAM-Based Buffering

**Why RAM instead of VRAM:**
- **Simpler:** Python/NumPy implementation (vs CUDA kernels)
- **Scalable:** 300MB for 50 streams (vs limited 8-24GB VRAM)
- **Latency acceptable:** ~1-2ms PCIe transfer (vs ~0.1ms VRAM, but 10-15ms inference dominates)
- **Development time:** 3-5 days (vs 2-3 weeks for CUDA)

### File to Create: `temporal_buffer.py`

**Location:** `/home/nick/Workspace/Drone-detection-dataset/temporal_buffer.py`

```python
import threading
from collections import deque, defaultdict
import numpy as np
import cv2

class TemporalFrameBuffer:
    """
    Thread-safe per-stream frame buffering for temporal YOLO inference.

    Stores last 5 frames per stream in CPU RAM.
    Memory: 6MB per stream (5 × 640×640×3 uint8).
    """

    def __init__(self, num_frames=5, img_size=640):
        self.num_frames = num_frames
        self.img_size = img_size
        self.buffers = defaultdict(lambda: deque(maxlen=num_frames))
        self.lock = threading.Lock()

    def add_frame(self, source_id: int, frame: np.ndarray):
        """
        Add frame to stream buffer.

        Args:
            source_id: Stream/camera ID
            frame: (H, W, 3) or (H, W, 4) uint8 array (RGB or RGBA)
        """
        # Resize and convert to RGB if needed
        if frame.shape[:2] != (self.img_size, self.img_size):
            frame = cv2.resize(frame, (self.img_size, self.img_size))

        if frame.shape[2] == 4:  # RGBA → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        with self.lock:
            self.buffers[source_id].append(frame.copy())

    def get_stacked_frames(self, source_id: int) -> np.ndarray:
        """
        Get 15-channel stacked frames for inference.

        Returns:
            (15, 640, 640) float32 array normalized to [0.0, 1.0]

        Cold start handling: Pads with first frame if buffer not full.
        """
        with self.lock:
            buffer = self.buffers[source_id]

            # Cold start: pad with first frame or zeros
            if len(buffer) < self.num_frames:
                if len(buffer) == 0:
                    # No frames yet - return zeros
                    return np.zeros((3 * self.num_frames, self.img_size, self.img_size),
                                   dtype=np.float32)
                # Pad with first frame
                frames = [buffer[0]] * (self.num_frames - len(buffer)) + list(buffer)
            else:
                frames = list(buffer)

            # Stack channel-wise: [t-4, t-3, t-2, t-1, t] → 15 channels
            stacked = np.empty((self.img_size, self.img_size, 3 * self.num_frames),
                              dtype=np.uint8)

            for i, frame in enumerate(frames):
                stacked[:, :, i*3:(i+1)*3] = frame

            # Convert to NCHW, normalize to [0, 1]
            stacked = stacked.transpose(2, 0, 1).astype(np.float32) / 255.0

            return stacked

    def reset(self, source_id: int):
        """Clear buffer for stream (e.g., on disconnect)."""
        with self.lock:
            if source_id in self.buffers:
                self.buffers[source_id].clear()
```

### Memory Footprint

```
1 frame = 640 × 640 × 3 = 1.2 MB
5 frames per stream = 6 MB

10 streams = 60 MB
50 streams = 300 MB
```

**Conclusion:** Extremely efficient - negligible RAM usage

---

## 5. DeepStream Pipeline Integration

### File to Create: `deepstream_temporal_inference.py`

**Location:** `/home/nick/Workspace/Drone-detection-dataset/deepstream_temporal_inference.py`

### Probe 1: Frame Capture (Before nvstreammux)

**Attach to:** `src` pad of `capsfilter` (RGBA output)

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import pyds
import numpy as np

# Global frame buffer
frame_buffer = TemporalFrameBuffer(num_frames=5, img_size=640)

def frame_buffer_probe(pad, info, user_data):
    """
    Captures frames from each stream and stores in RAM buffer.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            source_id = frame_meta.source_id

            # Get frame from NvBufSurface (GPU memory)
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_np = np.array(n_frame, copy=True, order='C')  # Copy to CPU

            # Add to buffer
            frame_buffer.add_frame(source_id, frame_np)

        except StopIteration:
            break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
```

### Probe 2: Temporal Stacking (Before nvinfer)

**Attach to:** `sink` pad of `nvinfer`

**Challenge:** DeepStream nvinfer expects 3-channel input by default. Need to use `nvdspreprocess` or custom buffer modification.

**Recommended approach:** Use `nvdspreprocess` element with custom transform function, OR modify nvinfer config to accept custom input preprocessing.

**Alternative (simpler but hacky):** Modify GstBuffer directly in probe:

```python
def temporal_stacking_probe(pad, info, user_data):
    """
    Replaces 3-channel frames with 15-channel stacked frames before inference.

    NOTE: This requires custom nvinfer configuration or nvdspreprocess element.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    frame_idx = 0

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            source_id = frame_meta.source_id

            # Get 15-channel stacked frames
            stacked = frame_buffer.get_stacked_frames(source_id)
            # Shape: (15, 640, 640) float32 [0.0, 1.0]

            # Modify NvBufSurface to contain 15-channel data
            # This may require custom nvdspreprocess plugin or direct buffer manipulation
            # See DeepStream SDK documentation for custom preprocessing

            frame_idx += 1
        except StopIteration:
            break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
```

**IMPORTANT:** The temporal stacking probe requires either:
1. **nvdspreprocess element** with custom transform function (recommended)
2. **Custom nvinfer preprocessing plugin** (more complex)
3. **Pre-batching stacking** before nvstreammux (simplest but requires pipeline redesign)

**Recommended Path:** Investigate `nvdspreprocess` element capabilities first. If it supports custom channel counts, use that. Otherwise, may need to batch frames CPU-side before muxer.

### Pipeline Configuration: `config_infer_temporal_yolo.txt`

**Location:** `/home/nick/Workspace/cluster-server-gstreamer/DeepStream-Yolo2/config_infer_temporal_yolo.txt`

**Based on:** Existing `config_infer_primary_yoloV8.txt`

**Key changes:**
```ini
[property]
gpu-id=0
net-scale-factor=1.0  # Normalization handled in preprocessing
model-color-format=0  # RGB (but actually 15-channel)
onnx-file=/path/to/yolov8n_temporal5.onnx
model-engine-file=/path/to/yolov8n_temporal5_fp16.engine
batch-size=4
network-mode=2  # FP16
num-detected-classes=1  # All classes mapped to single "airspace_object"
interval=0  # Process every frame

# Custom preprocessing may be needed here
# TODO: Investigate nvdspreprocess integration

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
```

---

## 6. Implementation Roadmap

### Phase 1: ONNX Export & TensorRT (2-3 days)

1. Create `export_temporal_yolo.py` (adapt from existing `export_yoloV8.py`)
   - Change input shape to `(batch, 15, 640, 640)`
   - Add validation for 15-channel first conv
2. Export trained model to ONNX
3. Create `test_onnx_export.py` validation script
4. Build TensorRT engine with `build_trt_engine.sh`
5. Profile with `trtexec` to measure inference time

**Success criteria:**
- ✅ ONNX file generated without errors
- ✅ PyTorch vs ONNX outputs match (tolerance 1e-4)
- ✅ TensorRT engine builds successfully
- ✅ Inference time <15ms (RTX 3080)

### Phase 2: Frame Buffering (1 day)

1. Create `temporal_buffer.py` with `TemporalFrameBuffer` class
2. Write unit tests:
   - Test add/retrieve frames
   - Test cold start handling (buffer not full)
   - Test thread safety (concurrent access)
3. Benchmark performance (<1ms retrieval)

**Success criteria:**
- ✅ All unit tests pass
- ✅ Thread-safe under load
- ✅ Memory usage: 6MB per stream

### Phase 3: DeepStream Integration (3-4 days)

**Challenge:** Getting 15-channel input into nvinfer

**Investigation tasks:**
1. Research `nvdspreprocess` custom transform capabilities
2. Check if nvinfer supports custom input preprocessing
3. Prototype alternative: batch stacking before muxer

**Implementation:**
1. Create `deepstream_temporal_inference.py`
2. Implement `frame_buffer_probe()` (frame capture)
3. Implement temporal stacking approach (based on investigation)
4. Test with single video file
5. Debug NvBufSurface handling
6. Handle edge cases (cold start, stream restart)

**Success criteria:**
- ✅ Probe callbacks execute without errors
- ✅ Frames correctly buffered and retrieved
- ✅ 15-channel data reaches TensorRT engine
- ✅ Single-stream inference produces detections

### Phase 4: Multi-Stream Testing (1-2 days)

1. Test with 2-4 streams
2. Stress test with 10+ streams
3. Monitor resources (CPU, RAM, GPU, VRAM)
4. Handle stream failures (disconnect/reconnect)
5. Measure per-stream FPS and latency

**Success criteria:**
- ✅ Pipeline handles 10+ streams
- ✅ Per-stream FPS >25 (real-time)
- ✅ No memory leaks over 1-hour run
- ✅ Graceful handling of stream disconnects

### Phase 5: Production Optimization (Optional, 2-3 days)

1. Profile bottlenecks (frame stacking, PCIe transfers)
2. Optimize with CUDA if needed (GPU-side frame stacking)
3. Add INT8 quantization for 2x speedup
4. Add monitoring/logging infrastructure
5. Package as Docker container

**Success criteria:**
- ✅ Frame stacking <0.5ms
- ✅ INT8 engine (optional) with <2% mAP loss
- ✅ Production-ready monitoring

---

## 7. Critical Files

### Files to Create

1. **`/home/nick/Workspace/Drone-detection-dataset/export_temporal_yolo.py`**
   - ONNX export for 15-channel temporal YOLO
   - Based on: `/home/nick/Workspace/cluster-server-gstreamer/DeepStream-Yolo2/utils/export_yoloV8.py`

2. **`/home/nick/Workspace/Drone-detection-dataset/test_onnx_export.py`**
   - Validate PyTorch vs ONNX outputs

3. **`/home/nick/Workspace/Drone-detection-dataset/build_trt_engine.sh`**
   - TensorRT engine build script

4. **`/home/nick/Workspace/Drone-detection-dataset/temporal_buffer.py`**
   - Per-stream frame buffering class

5. **`/home/nick/Workspace/Drone-detection-dataset/deepstream_temporal_inference.py`**
   - GStreamer pipeline with probe callbacks

6. **`/home/nick/Workspace/cluster-server-gstreamer/DeepStream-Yolo2/config_infer_temporal_yolo.txt`**
   - nvinfer configuration for temporal YOLO

### Files to Reference

7. **`/home/nick/Workspace/Drone-detection-dataset/training/model.py`**
   - `get_first_conv()` - validation helper
   - `expand_first_conv()` - understanding weight initialization

8. **`/home/nick/Workspace/Drone-detection-dataset/training/data.py:140-149`**
   - Frame stacking logic to replicate in probe

9. **`/home/nick/Workspace/cluster-server-gstreamer/DeepStream-Yolo2/utils/export_yoloV8.py`**
   - Template for ONNX export

---

## 8. Key Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| TensorRT rejects 15-channel conv | 10% | High | Test early; fallback to ONNX Runtime |
| nvdspreprocess limitations | 30% | Medium | Alternative: pre-mux batching |
| Python probe performance | 20% | Medium | Profile early; rewrite in C++ if needed |
| VRAM exhaustion (multi-stream) | 15% | Medium | Dynamic batching, INT8 |

**Highest Priority Risk:** nvdspreprocess/nvinfer 15-channel input handling - investigate this FIRST before full implementation.

---

## 9. Success Metrics

**Functional:**
- ✅ Pipeline runs for 1+ hour without crashes
- ✅ Correct detections on test videos (visual validation)
- ✅ All streams processed independently (no cross-talk)

**Performance:**
- ✅ >25 FPS per stream (real-time @ 30fps)
- ✅ 10+ simultaneous streams on RTX 3080
- ✅ <25ms end-to-end latency (33ms budget @ 30fps)

**Accuracy:**
- ✅ Temporal mAP ≥ standard YOLO mAP (motion benefits)
- ✅ No significant accuracy degradation from ONNX/TRT conversion

**Memory:**
- ✅ RAM usage scales linearly (6MB per stream)
- ✅ VRAM usage <2GB (model + activations)

---

## 10. Next Steps

**Before Implementation:**
1. **Investigate nvdspreprocess** - Can it handle 15-channel input? This is critical path.
2. **Profile TensorRT build** - Ensure 15-channel conv compiles without errors
3. **Review DeepStream SDK docs** - Custom preprocessing examples

**First Implementation Task:**
Export ONNX and build TensorRT engine - this validates the core feasibility.

**If nvdspreprocess doesn't support 15 channels:**
Alternative approach: Batch 5 consecutive frames on CPU, stack them, then feed to custom GStreamer element or direct TensorRT inference (bypass DeepStream nvinfer).
