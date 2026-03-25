# Architecture Audit — `demo_detection_realtime`

## 1. CURRENT ARCHITECTURE

### 1.1 Frame Flow: Reader → Detector → Compositor → Flask Stream

```
┌──────────────────────────────────────────────────────────────┐
│ Thread 1: _reader_loop  (VideoReader)                        │
│  cv2.VideoCapture.read() → rotate → store raw frame          │
│     ├─ _raw_frame   (latest, under _raw_lock)                │
│     ├─ _raw_history  (deque, maxlen=24, under _raw_history_lock) │
│     └─ _det_frame   (copy, 1-in-N, under _det_lock)          │
│         signals _det_event                                    │
└───────────┬──────────────────────────────┬───────────────────┘
            │ _det_event                   │ _raw_changed
            ▼                              ▼
┌──────────────────────────┐  ┌─────────────────────────────────┐
│ Thread 2: _detection_loop│  │ Thread 3: _compositor_loop       │
│  (YOLODetector)          │  │  (Compositor)                    │
│  YOLO .track()/.predict()│  │  raw_frame + overlay → annotate  │
│  → builds overlay dict   │  │  → cv2.imencode JPEG             │
│  stores in _overlay      │  │  stores in _jpeg_bytes           │
│  (under _overlay_lock)   │  │  (under _jpeg_lock)              │
└──────────────────────────┘  └──────────┬──────────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────────┐
                              │ Flask /video_feed         │
                              │  yields _jpeg_bytes       │
                              │  sleep(0.066) ≈ 15 fps    │
                              └──────────────────────────┘
```

**Reader** reads at native FPS (video file: throttled by `time.sleep`; camera: limited by hardware).
It stores every frame in `_raw_frame` (latest-only),
pushes a copy to `_raw_history` (24-frame ring buffer),
and every Nth frame (`DETECTOR_FRAME_SKIP`, default=1, i.e. **every** frame) copies to `_det_frame` + signals `_det_event`.

**Detector** wakes on `_det_event`, grabs `_det_frame`, runs YOLO, builds an overlay dict (boxes, labels, colors), stores it under `_overlay_lock`.

**Compositor** wakes on `_raw_changed` (or 100 ms timeout), grabs the latest raw frame, looks up `_raw_history` for the exact detector frame (to avoid visual box shift), draws boxes/lines/HUD on top, then `cv2.imencode('.jpg')` with quality 80. The JPEG bytes go into `_jpeg_bytes`.

**Flask `/video_feed`** is a generator that reads `_jpeg_bytes` under `_jpeg_lock` and yields it as MJPEG, polling at `time.sleep(0.066)` → ~15 HTTP fps cap regardless of pipeline speed.

### 1.2 Model Switching (Tracking / Date / Segmentation+Anomaly)

There are 4 checkpoints defined in `tracking_config.py`:

| ID            | Mode       | Primary Model                 | Secondary Model        |
|---------------|------------|-------------------------------|------------------------|
| `tracking`    | tracking   | `bestexp2.pt`                 | None                   |
| `date`        | date       | `yolo26-BB(date).pt`          | None                   |
| `barcode_date`| tracking   | `yolo26m_BB_barcode_date.pt`  | `yolo26-BB(date).pt`   |
| `anomaly`     | anomaly    | `yolo26m_seg_farine_FV.pt`    | None (uses EfficientAD)|

Switching is handled by `TrackingState.switch_checkpoint()`:
1. `stop_processing()` — sets `is_running=False`, waits 0.5 s, releases cap
2. Deletes primary model, secondary model, and EfficientAD models from VRAM; calls `torch.cuda.empty_cache()` + `gc.collect()`
3. Loads new YOLO model via `YOLO(path).to(DEVICE)` where `DEVICE='cuda'`
4. Resolves class IDs by scanning `model.names`
5. If mode is `tracking` and `secondary_date_model_path` is set, loads secondary YOLO model
6. If mode is `anomaly`, calls `_load_ad_models()` to load teacher/student/autoencoder `.pth` files
7. Warmup: dummy inference on a 640×640 zeros tensor, then `torch.cuda.empty_cache()`
8. Returns status dict; caller (`api_switch`) restarts processing if it was running

The web server also has `init_models()` called at startup, which does the same loading but via the `YOLO()` constructor + `.to(DEVICE)`.

### 1.3 Thread/Process Spawning and Management

**All concurrency is `threading.Thread(daemon=True)`** — no multiprocessing.

Threads spawned per session (in `start_processing` or `resume_processing`):
1. `VideoReader` — `_reader_loop(session_gen)`
2. `YOLODetector` — `_detection_loop()`
3. `Compositor` — `_compositor_loop()`

Additional threads:
4. `DBWriter` — background writer thread (started lazily on first stats toggle), runs `_run()` loop draining `write_queue`.
5. Per-NOK-packet: `_save_nok_packet_bg()` spawns a fire-and-forget daemon thread `AD-Save-{pkt_num}` to write crop images + CSV to disk.

**Session generation** (`_session_gen`): an integer counter bumped before launching threads. Only the reader thread receives `session_gen`; on exit, it only clears `is_running` if `_session_gen` still matches. Detector/compositor check `self.is_running` to know when to stop.

**Lifecycle:**
- `stop_processing()` sets `is_running=False`, sleeps 0.5 s, releases cap, frees CUDA cache.
- `pause_processing()` saves `cap.get(POS_FRAMES)`, calls `stop_processing()` equivalent, sets `_is_paused=True`.
- `resume_processing()` does a partial reset (preserves accumulated stats), relaunches all 3 threads.

There is **no join** on detector/compositor threads — they're daemon threads that exit when `is_running` becomes False.

Flask itself runs in **threaded mode** (`app.run(threaded=True)`), so each HTTP request is served in its own thread (Werkzeug's `ThreadingMixIn`).

### 1.4 Frame Queues and Max Sizes

| Queue / Buffer                | Type                          | Max Size    | Protection           |
|-------------------------------|-------------------------------|-------------|----------------------|
| `_raw_frame`                  | Single latest frame           | 1 (latest)  | `_raw_lock` (Lock)   |
| `_raw_history`                | `deque(maxlen=24)`            | 24 frames   | `_raw_history_lock`  |
| `_det_frame`                  | Single latest frame           | 1 (latest)  | `_det_lock` (Lock)   |
| `_jpeg_bytes`                 | Single JPEG blob              | 1 (latest)  | `_jpeg_lock` (Lock)  |
| `_overlay`                    | Single overlay dict           | 1 (latest)  | `_overlay_lock`      |
| `DBWriter.write_queue`        | `queue.Queue`                 | 10000       | Thread-safe Queue    |
| `cv2.VideoCapture` buffer     | Set to 1 for live/RTSP        | 1 frame     | (OpenCV internal)    |

**There are no bounded producer-consumer queues between reader/detector/compositor.** The design is "latest frame wins" — detector always processes the newest available frame, compositor always draws on the newest raw frame.

---

## 2. SEGMENTATION + ANOMALY PIPELINE (mode="anomaly")

### 2.1 Exact Steps: Raw Frame → Final Annotated Output

```
Raw frame (BGR, full resolution, e.g. 1280×720)
  │
  ├─ 1. Reader: rotate (optional), store in _det_frame
  │
  ▼
Detection loop picks up _det_frame
  │
  ├─ 2. YOLO segmentation + tracking:
  │     model.track(frame, conf=0.5, imgsz=640, retina_masks=True,
  │                 persist=True, tracker=bytetrack.yaml)
  │     Returns: results.masks.data  (GPU tensor, lower-res mask per instance)
  │              results.boxes.xyxy  (GPU tensor, bounding boxes)
  │              results.boxes.id    (GPU tensor, track IDs)
  │
  ├─ 3. Transfer to CPU:
  │     masks = results.masks.data.cpu().numpy()      # float32 ndarray
  │     boxes = results.boxes.xyxy.cpu().numpy()       # float32 ndarray
  │     track_ids = results.boxes.id.int().cpu().tolist()  # list[int]
  │
  ├─ 4. Per-tracked-object:
  │     │
  │     ├─ Zone check (center_x vs entry/exit lines) ──────────────────┐
  │     │    center_x > entry_line → ENTERING (skip)                    │
  │     │    exit <= center_x <= entry → SCAN ZONE                      │
  │     │    center_x < exit → DECISION (lock final result, append FIFO)│
  │     │                                                               │
  │     ├─ If SCAN ZONE and n_scans < ad_max_scans (default 5):        │
  │     │   │                                                           │
  │     │   ├─ 5. _ad_crop_and_mask(frame, masks[i], checkpoint):      │
  │     │   │     a) cv2.resize mask to full frame resolution           │
  │     │   │     b) Threshold at 0.5                                   │
  │     │   │     c) Find bounding box of mask pixels (np.where)        │
  │     │   │     d) Expand by margin_pct (default 10%)                 │
  │     │   │     e) Crop BGR frame to bounding box                     │
  │     │   │     f) GaussianBlur + threshold + erode mask (CPU)        │
  │     │   │     g) Blackout background: img_crop[mask==0] = 0         │
  │     │   │     h) Convert BGR→RGB                                    │
  │     │   │     i) letterbox_image(img_rgb) → 640×640 padded (CPU)    │
  │     │   │     Returns: numpy array (H, W, 3), uint8, RGB            │
  │     │   │                                                           │
  │     │   ├─ 6. _ad_detect_anomaly(img_crop_np):                     │
  │     │   │     a) PIL.Image.fromarray(crop)                          │
  │     │   │     b) torchvision transforms: Resize(256×256), ToTensor, │
  │     │   │        Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) │
  │     │   │     c) .unsqueeze(0).to(device)  → GPU tensor [1,3,256,256]│
  │     │   │     d) efficientad.predict() — teacher/student/autoencoder│
  │     │   │        forward passes (all on GPU, @torch.no_grad)        │
  │     │   │     e) F.pad + F.interpolate map back to original crop size│
  │     │   │     f) map_combined[0,0].cpu().numpy().max() → scalar score│
  │     │   │     g) Compare score > ad_thresh (default 5000.0)         │
  │     │   │     Returns: (is_defective: bool, score: float)           │
  │     │   │                                                           │
  │     │   └─ Store result in tstate['results'], score, crop copy      │
  │     │                                                               │
  │     └─ If DECISION ZONE:                                            │
  │         _ad_final_decision(results, strategy="MAJORITY")            │
  │         Append OK/NOK to output_fifo                                │
  │         If NOK: _save_nok_packet_bg() → background thread saves     │
  │                 crop PNGs + CSV to anomalie/<pkt_num>/              │
  │                                                                     │
  ├─ 5. Build overlay dict with track_boxes + ad_zone_lines             │
  │                                                                     │
  ▼                                                                     │
Compositor draws boxes, zone lines, HUD → JPEG encode                  │
```

### 2.2 Where the Anomaly Model Receives Input from Segmentation

The YOLO segmentation model produces **instance masks** (`results.masks.data` — a GPU tensor of shape `[N, mask_h, mask_w]` where `mask_h/mask_w` are typically the YOLO input size, not the original frame size).

These masks are transferred to CPU via `.cpu().numpy()` in the detector loop (line ~1190 of `tracking_state.py`).

For each tracked instance in the scan zone, `_ad_crop_and_mask()` takes:
- **The original full-resolution BGR frame** (not the YOLO-resized frame)
- **One instance mask** (low-resolution float array)

It **resizes the mask** up to match the full frame, then uses it to crop and background-blackout the object, producing a clean isolated object image that is then **letterboxed to 640×640** (via `letterbox_image`).

This 640×640 RGB numpy array is then fed to `_ad_detect_anomaly()`, which resizes it again to **256×256** via torchvision transforms before passing to EfficientAD.

### 2.3 How Masks Are Processed Between Models

1. **YOLO outputs masks** as a GPU float tensor at model-internal resolution (not frame resolution). `retina_masks=True` requests higher-quality masks from Ultralytics.
2. **`.cpu().numpy()`** transfers all masks to CPU in one batch operation.
3. **Per-instance** in `_ad_crop_and_mask()`:
   - `cv2.resize(mask, (frame_w, frame_h))` — upscales to full frame resolution (CPU, INTER_LINEAR)
   - Threshold at 0.5 → binary
   - `np.where(mask > 0)` to find bounding box
   - Crop + margin expansion
   - Optional: GaussianBlur(5,5) → threshold → erode (morphological cleanup, CPU)
   - Background blackout: `img_crop[mask_final == 0] = 0`

### 2.4 Blocking Calls / Synchronous Waits

The anomaly pipeline has **multiple synchronous blocking points** inside the detector thread:

| Blocking Call | Location | Nature |
|---|---|---|
| `model.track()` | Detection loop | GPU inference, synchronous — blocks until YOLO finishes segmentation+tracking |
| `results.masks.data.cpu().numpy()` | Detection loop | GPU→CPU transfer, blocks until CUDA stream completes |
| `results.boxes.xyxy.cpu().numpy()` | Detection loop | GPU→CPU transfer |
| `cv2.resize(mask, ...)` | `_ad_crop_and_mask` | CPU operation, per-instance |
| Mask morphology (GaussianBlur, erode) | `_ad_crop_and_mask` | CPU operations |
| `letterbox_image()` | `_ad_crop_and_mask` | CPU cv2.resize + numpy allocation |
| `PIL.Image.fromarray()` | `_ad_detect_anomaly` | CPU copy |
| `self._ad_transform(pil_img)` | `_ad_detect_anomaly` | CPU Resize(256)+ToTensor+Normalize |
| `img_tensor.to(device)` | `_ad_detect_anomaly` | CPU→GPU transfer |
| `effpredict()` (teacher+student+autoencoder forward) | `_ad_detect_anomaly` | GPU inference, synchronous |
| `map_combined[0,0].cpu().numpy()` | `_ad_detect_anomaly` | GPU→CPU transfer for score |

**All of these run sequentially in the same detector thread, for each tracked instance, on every frame where the instance is in the scan zone.** There is no batching — each object is processed one at a time.

If there are K objects in the scan zone on a single frame, the detector performs:
- 1× YOLO segmentation+tracking inference
- K× mask crop+cleanup (CPU)
- K× EfficientAD inference (GPU, serial)

---

## 3. BOTTLENECK ANALYSIS

### 3.1 CPU vs GPU Function Breakdown

| Function / Operation | Device | Notes |
|---|---|---|
| `cv2.VideoCapture.read()` | CPU | Video decoding (may use HW accel) |
| `_rotate_frame_ccw()` | CPU | `cv2.rotate()` |
| `frame.copy()` (reader→detector) | CPU | Full frame copy every N frames |
| `model.track()` / `model()` | **GPU** | YOLO inference |
| `.cpu().numpy()` on results | CPU←GPU | Synchronous transfer, implicit sync |
| `cv2.resize(mask, ...)` | CPU | Upscale mask to frame res |
| `np.where(mask > 0)` | CPU | Bounding box from mask |
| `GaussianBlur`, `threshold`, `erode` | CPU | Mask morphology |
| `img_crop[mask==0] = 0` | CPU | Numpy masked assignment |
| `cv2.cvtColor(BGR→RGB)` | CPU | Color conversion |
| `letterbox_image()` | CPU | cv2.resize + numpy pad |
| `PIL.Image.fromarray()` | CPU | Array→PIL copy |
| `torchvision.transforms (Resize, ToTensor, Normalize)` | CPU | Preprocessing for EfficientAD |
| `.unsqueeze(0).to(device)` | CPU→**GPU** | Transfer preprocessed tensor |
| `efficientad.predict()` (teacher/student/autoencoder) | **GPU** | 3 forward passes under `@torch.no_grad` |
| `F.pad`, `F.interpolate` | **GPU** | Post-process anomaly map |
| `map[0,0].cpu().numpy().max()` | CPU←GPU | Transfer anomaly map, compute max |
| Compositor: all `cv2.rectangle`, `cv2.putText`, `cv2.line` | CPU | OpenCV drawing |
| `cv2.imencode('.jpg', ...)` | CPU | JPEG encoding |

### 3.2 Torch Operations and Device Placement

All torch models are loaded to `DEVICE = 'cuda'`:
- **Primary YOLO model:** `YOLO(path).to('cuda')` — inference via `.track()` / `.__call__()` runs on GPU
- **Secondary YOLO model** (barcode_date checkpoint): `.to('cuda')`, inference via `.__call__()` on GPU
- **EfficientAD teacher/student/autoencoder:** `torch.load(..., map_location='cuda').eval()` — all 3 models loaded to GPU

The `efficientad.predict()` function (decorated with `@torch.no_grad()`) runs `teacher(image)`, `student(image)`, `autoencoder(image)` — all on whichever device the models sit on (GPU). The output tensor processing (subtraction, mean, pad, interpolate) also stays on GPU until the final `.cpu().numpy()` call to extract the scalar score.

### 3.3 `.cpu()` / `.numpy()` Conversions Mid-Pipeline

These are the synchronization-forcing transfers that happen **inside the hot detection loop**:

**Tracking/Date mode (per frame):**
1. `b.xyxy[0].cpu().numpy()` — per-box in the tracking result extraction loop (individual box, not batched)
2. `b.cls`, `b.conf` — accessed as Python scalars (implicit `.item()`)

**Anomaly mode (per frame + per object):**
1. `results.masks.data.cpu().numpy()` — all masks at once
2. `results.boxes.xyxy.cpu().numpy()` — all boxes at once
3. `results.boxes.id.int().cpu().tolist()` — all track IDs
4. Per-object in scan zone: `map_combined[0,0].cpu().numpy().max()` — anomaly map

**Tracking mode with secondary date model (per frame, additional):**
1. `b.xyxy[0].cpu().numpy()` — per-box from secondary model results

**Key observation:** In tracking mode, box extraction does `.cpu().numpy()` **per individual box** inside a Python loop. In anomaly mode, boxes and masks are batched into a single `.cpu().numpy()` call but then iterated on CPU.

### 3.4 Thread Contention Points

| Contention | Locks Involved | Impact |
|---|---|---|
| Reader writes `_raw_frame`, compositor reads it | `_raw_lock` | Minimal — fast pointer swap |
| Reader writes `_raw_history`, compositor reads for frame matching | `_raw_history_lock` | Low — deque append vs reverse search |
| Reader writes `_det_frame`, detector reads it | `_det_lock` | Minimal — copy on write, read once |
| Detector writes `_overlay`, compositor reads it | `_overlay_lock` | Low — dict swap |
| Compositor writes `_jpeg_bytes`, Flask reads it | `_jpeg_lock` | Low — bytes swap |
| Flask reads `_stats_lock` for stats, detector writes | `_stats_lock` | Low — dict update |
| DBWriter `write_queue` | `queue.Queue` internal | Non-blocking `put_nowait`; 10000 capacity |
| GIL (Python Global Interpreter Lock) | All threads | **Significant**: all threads share the GIL. CPU-bound work (OpenCV, numpy, frame copies) in the compositor and reader compete with the detector thread. However, GPU inference releases the GIL while waiting for CUDA kernels. |

**GIL impact specifics:**
- `cv2.imencode()` in compositor: holds GIL during JPEG encoding (~1–5 ms for 720p)
- `frame.copy()` in reader: holds GIL during memory copy
- `cv2.resize()` in mask processing: holds GIL
- Numpy operations in mask processing: many release GIL, some don't
- `torch.no_grad()` GPU ops: release GIL while CUDA kernel runs

---

## 4. CURRENT PERFORMANCE METRICS (from code)

### 4.1 Frame Queue Sizes

| Buffer | Size | Drop Policy |
|---|---|---|
| `_raw_frame` | 1 (latest-only) | Overwritten by reader every frame |
| `_raw_history` | 24 frames (deque) | Oldest dropped automatically |
| `_det_frame` | 1 (latest-only) | Overwritten; detector skips any missed frames |
| `_jpeg_bytes` | 1 (latest-only) | Overwritten by compositor |
| `write_queue` (DB) | 10,000 events | `put_nowait` — silently drops if full |
| OpenCV capture buffer | 1 (set explicitly for live/RTSP) | OpenCV drops old frames |

### 4.2 Frame Skipping Logic

**Reader → Detector skip:**
```python
# tracking_config.py
DETECTOR_FRAME_SKIP = 1   # feed every frame to detector
```
The reader only posts to `_det_frame` when `frame_count % DETECTOR_FRAME_SKIP == 0`. With the default of 1, every frame goes to the detector.

**Detector internal skip:**
The detector tracks `last_processed_idx` and skips frames whose `frame_idx <= last_processed_idx`. Since the reader can produce frames faster than the detector processes them, the detector naturally processes only the **most recent** available frame, effectively skipping all intermediate frames that accumulated while it was busy.

**Video file FPS throttling:**
For video files, the reader sleeps to match native FPS: `time.sleep(frame_time - elapsed)`. This limits how fast frames feed in.

**Flask MJPEG throttle:**
`time.sleep(0.066)` in the `/video_feed` generator caps the HTTP stream at ~15 fps.

**DB snapshot throttle:**
Snapshots are only enqueued every `SNAPSHOT_EVERY_N_PACKETS = 25` packets.

### 4.3 `imgsz` Values Per Model

| Model | imgsz | Source |
|---|---|---|
| Primary YOLO (tracking, date, barcode_date) | 640 | `CONFIG["imgsz"] = 640` |
| Primary YOLO (anomaly) | 640 | `checkpoint["yolo_imgsz"] = 640` |
| Secondary YOLO (date, in barcode_date checkpoint) | 640 | `CONFIG["imgsz"]` |
| EfficientAD (teacher/student/autoencoder) | 256 | `checkpoint["ad_imgsz"] = 256` |
| Letterbox pre-EfficientAD crop | 640 | `letterbox_image(img, size=640)` default |
| YOLO warmup dummy | 640 | Hardcoded `np.zeros((640, 640, 3))` |

**Note:** The anomaly pipeline letterboxes the crop to **640×640** first, then the torchvision transform resizes to **256×256** for EfficientAD. This is a double resize.

### 4.4 Number of Threads Spawned

| Thread | Count | Lifetime |
|---|---|---|
| `VideoReader` | 1 per session | Exits with session |
| `YOLODetector` | 1 per session | Exits with session |
| `Compositor` | 1 per session | Exits with session |
| `DBWriter` | 0 or 1 (lazy) | Lives until app shutdown |
| `AD-Save-{N}` | 1 per NOK packet | Fire-and-forget, exits after disk I/O |
| Flask worker threads | N (Werkzeug ThreadingMixin) | Per-request, pooled |

**Total during anomaly detection with active recording:** 3 (core) + 1 (DB writer) + M (save threads for M NOK packets being written) + Flask threads.

During checkpoint switch, old threads are not joined — they exit via `is_running=False` check. New threads are launched immediately. The session generation counter prevents old reader threads from incorrectly resetting `is_running`.
