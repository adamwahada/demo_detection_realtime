# System Architecture — Full Technical Report

**Date:** 2026-03-31  
**Purpose:** Complete description of how the backend works, every thread, every connection, the full request/data flow.

---

## 1. Process Overview

One single Python process (`python web_server_backend_v2.py`) runs everything:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Python Process (PID)                                 │
│                                                                             │
│  ┌─────────────────┐   ┌─────────────────┐   ┌───────────────────────────┐ │
│  │  gevent WSGI    │   │  APScheduler    │   │  DBWriter                 │ │
│  │  Server         │   │  (BackgroundSch)│   │  (write thread)           │ │
│  │  (greenlets)    │   │  1 OS thread    │   │  1 OS thread              │ │
│  └─────────────────┘   └─────────────────┘   └───────────────────────────┘ │
│                                                                             │
│  ┌──────────────────────────┐  ┌──────────────────────────────────────────┐ │
│  │  pipeline_0 (tracking)   │  │  pipeline_1 (anomaly)                   │ │
│  │  3 OS threads:           │  │  3 OS threads:                          │ │
│  │    Reader                │  │    Reader                               │ │
│  │    Detector (YOLO+Track) │  │    Detector (YOLO+EfficientAD)          │ │
│  │    Compositor            │  │    Compositor                           │ │
│  └──────────────────────────┘  └──────────────────────────────────────────┘ │
│                                                                             │
│  ┌──────────────────────────┐  ┌──────────────────────────────────────────┐ │
│  │  _secondary_executor     │  │  _proof_executor                        │ │
│  │  ThreadPoolExecutor(1)   │  │  ThreadPoolExecutor(4)                  │ │
│  │  1 OS worker thread      │  │  4 OS worker threads                    │ │
│  └──────────────────────────┘  └──────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Startup Sequence

File: `web_server_backend_v2.py`, `if __name__ == '__main__':` block (line ~1750)

### Step 1: Monkey-Patching (line 1-2)
```python
from gevent import monkey
monkey.patch_all(thread=False, queue=False)
```
- **What it patches:** `socket`, `time.sleep`, `select`, `ssl`, `os`, `signal`, `subprocess`
- **What it does NOT patch:** `threading` (thread=False), `queue` (queue=False)
- **Why:** PyTorch/CUDA use real OS threads internally. Patching `threading` replaces them with greenlets → segfault. Patching `queue` breaks `concurrent.futures.ThreadPoolExecutor` which uses `queue.SimpleQueue` in its worker threads.

### Step 2: DBWriter Init
```python
db_writer = DBWriter() if _DB_AVAILABLE else None
```
- Tries to connect to PostgreSQL (`localhost:5434`, db=`farine_detection`)
- If PG is reachable → backend = "postgres", creates tables/indexes if missing
- If PG unreachable → falls back to SQLite at `data/tracking_demo.db`
- Opens SQLite write connection + read connection (WAL mode) if SQLite backend
- PG pool (`ThreadedConnectionPool`) is lazily created on first `_get_pg_conn()` call

### Step 3: Pipeline Init (`init_all_pipelines`)
For each entry in `PIPELINES` config:

**pipeline_0** (tracking mode):
1. Creates `TrackingState(pipeline_id="pipeline_0", db_writer=db_writer)`
2. Loads YOLO model `yolo26m_BB_barcode_date.pt` → GPU (cuda)
3. Resolves class IDs: package=2, barcode=0, date=1
4. Loads secondary date model `yolo26-BB(date).pt` → GPU
5. Runs dummy inference (warmup) to compile CUDA kernels

**pipeline_1** (anomaly mode):
1. Creates `TrackingState(pipeline_id="pipeline_1", db_writer=db_writer)`
2. Loads YOLO segmentation model `yolo26m_seg_farine_FV.pt` → GPU
3. Loads EfficientAD models (teacher, student, autoencoder) → GPU
4. Runs warmup on both YOLO and EfficientAD

### Step 4: Shift Scheduler
```python
_load_all_shift_jobs()
scheduler.start()
```
- `scheduler` = `BackgroundScheduler()` from APScheduler — runs in **1 background OS thread**
- Loads shift definitions from DB (`shifts` table)
- For each recurring shift: creates 2 CronTrigger jobs (_shift_start at start_time, _shift_stop at end_time)
- For each one-off shift: creates 2 DateTrigger jobs (exact datetime)
- Timezone: all triggers use `Africa/Tunis` (UTC+1)

### Step 5: HTTP Server
```python
from gevent.pywsgi import WSGIServer
http_server = WSGIServer((SERVER_HOST, SERVER_PORT), app, log=None)
http_server.serve_forever()
```
- gevent WSGIServer uses an **event loop** (libev/libuv) — NOT OS threads
- Each incoming HTTP request runs as a **greenlet** (coroutine)
- Greenlets yield cooperatively during I/O (socket read/write) and `time.sleep()`
- The MJPEG `/video_feed` generator's `while True` loop with `time.sleep(0.066)` yields to other greenlets during sleep

---

## 3. Thread Inventory (When Pipelines Are Running)

| # | Thread Name | Type | Source | Purpose |
|---|-------------|------|--------|---------|
| 1 | MainThread | OS thread | Python | gevent event loop + all HTTP greenlets |
| 2 | APScheduler | OS thread | `BackgroundScheduler` | Checks cron triggers, fires `_shift_start`/`_shift_stop` |
| 3 | DBWriter | OS thread | `db_writer.start()` | Drains `write_queue`, executes INSERT/UPDATE |
| 4 | VideoReader (p0) | OS thread (daemon) | `_reader_loop` | Reads frames from video/camera at native FPS |
| 5 | YOLODetector (p0) | OS thread (daemon) | `_detection_loop` | YOLO + ByteTrack inference + tracking logic |
| 6 | Compositor (p0) | OS thread (daemon) | `_compositor_loop` | Composites overlay onto frame → JPEG encode |
| 7 | VideoReader (p1) | OS thread (daemon) | `_reader_loop` | Reads frames from video/camera |
| 8 | YOLODetector (p1) | OS thread (daemon) | `_detection_loop` | YOLO segmentation + EfficientAD anomaly |
| 9 | Compositor (p1) | OS thread (daemon) | `_compositor_loop` | Composites overlay + zone lines → JPEG |
| 10 | SecModel-0 | OS thread (daemon) | `_secondary_executor` | Idle; wakes when secondary date model submitted |
| 11-14 | ProofSave-0..3 | OS threads (daemon) | `_proof_executor` | Idle; wake for proof-image PNG/CSV saves |

**Total: ~14 OS threads** when both pipelines are running.

Plus **N greenlets** in the main thread for active HTTP connections (lightweight, cooperative).

---

## 4. Per-Pipeline Thread Architecture

Each `TrackingState` has 3 threads that communicate via shared memory + events:

```
                ┌──────────────────────┐
                │   Video Source       │
                │  (file / camera /    │
                │   RTSP stream)       │
                └──────────┬───────────┘
                           │ cv2.VideoCapture.read()
                           ▼
                ┌──────────────────────┐
                │  THREAD 1: Reader    │
                │  ─────────────────── │
                │  • Reads at native   │
                │    FPS (30fps)       │
                │  • Rotates frame if  │
                │    rotation enabled  │
                │  • Stores in:        │
                │    _raw_frame        │  ──→ Compositor reads this
                │    _raw_history[24]  │  ──→ Compositor matches overlay to correct frame
                │  • Every N frames:   │
                │    _det_frame        │  ──→ Detector reads this
                │    signals _det_event│
                │  • Sleeps to match   │
                │    FPS (video files) │
                └──────────────────────┘
                           │
              _det_event.set() (every DETECTOR_FRAME_SKIP frames)
                           │
                           ▼
                ┌──────────────────────┐
                │  THREAD 2: Detector  │
                │  ─────────────────── │
                │  • Waits on          │
                │    _det_event        │
                │  • YOLO .track()     │
                │    or .predict()     │
                │  • ByteTrack assigns │
                │    track IDs         │
                │  • Extracts:         │
                │    - package tracks  │
                │    - barcode dets    │
                │    - date dets       │
                │  • Exit line cross   │
                │    detection →       │
                │    OK/NOK decision   │
                │  • Writes to:        │
                │    _overlay dict     │  ──→ Compositor reads this
                │    stats dict        │  ──→ Flask API reads this
                │  • Enqueues DB       │
                │    events to         │
                │    write_queue       │  ──→ DBWriter thread reads this
                └──────────────────────┘
                           │
                 _overlay updated, _raw_changed still set
                           │
                           ▼
                ┌──────────────────────┐
                │  THREAD 3: Compositor│
                │  ─────────────────── │
                │  • Waits on          │
                │    _raw_changed      │
                │  • Reads _raw_frame  │
                │    (or matched frame │
                │    from _raw_history)│
                │  • Reads _overlay    │
                │  • Draws:            │
                │    - bounding boxes  │
                │    - exit line       │
                │    - labels          │
                │    - FPS counter     │
                │    - FIFO queue text │
                │  • cv2.imencode JPEG │
                │  • Stores in         │
                │    _jpeg_bytes       │  ──→ /video_feed reads this
                └──────────────────────┘
```

### Shared Memory Between Threads (per pipeline)

| Variable | Written by | Read by | Lock |
|----------|-----------|---------|------|
| `_raw_frame` | Reader | Compositor | `_raw_lock` |
| `_raw_history` | Reader | Compositor | `_raw_history_lock` |
| `_det_frame` | Reader | Detector | `_det_lock` |
| `_det_event` | Reader | Detector | threading.Event (no lock) |
| `_raw_changed` | Reader | Compositor | threading.Event (no lock) |
| `_overlay` | Detector | Compositor | `_overlay_lock` |
| `_jpeg_bytes` | Compositor | `/video_feed` greenlet | `_jpeg_lock` |
| `stats` | Detector | `/api/stats` greenlet | `_stats_lock` |
| `_perf` | Detector + Compositor | `/api/perf` greenlet | `_perf_lock` |
| `write_queue` | Detector | DBWriter thread | `queue.Queue` (thread-safe) |

---

## 5. Detection Modes

### Mode: "tracking" (pipeline_0)

1. YOLO `.track(frame, persist=True, tracker=bytetrack.yaml)` → boxes + track IDs
2. Extract per class: packages (cls=2), barcodes (cls=0), dates (cls=1)
3. **Secondary date model** (parallel): submit `secondary_model(frame)` to `_secondary_executor` → `Future`
4. While waiting: process primary results (assign barcodes/dates to packages by IoU)
5. Collect `sec_future.result(timeout=2.0)` → merge + deduplicate date detections
6. Exit line crossing: when package bbox crosses line → lock decision (OK/NOK)
7. If NOK: submit proof-image save to `_proof_executor`
8. Every 25 packets: enqueue `session_update` to `write_queue`
9. Every NOK: enqueue `crossing` event to `write_queue`

### Mode: "anomaly" (pipeline_1)

1. YOLO `.track(frame, retina_masks=True)` → segmentation masks + track IDs
2. Three zones based on horizontal position (% of frame width):
   - **Entering** (right of entry_line): just tracking, no scanning
   - **Scanning** (between exit_line and entry_line): crop + mask → letterbox → EfficientAD batched inference
   - **Exiting** (left of exit_line): final MAJORITY decision → OK/NOK
3. EfficientAD runs batched: all crops in one forward pass through teacher/student/autoencoder
4. Per-track state tracks scan results (`_ad_track_states[tid]`)
5. NOK packets: save worst crop + CSV to `liveImages/<session>/anomalie/<pkt_num>/`

### Mode: "date" (not configured in PIPELINES, but switchable)

1. Plain YOLO detection (no tracking)
2. Overlay-only: draws all detected date boxes
3. No exit-line logic, no OK/NOK

---

## 6. Database Architecture

### Backend Selection (at startup)

```
psycopg2 importable?
  ├── YES → Try connect to localhost:5434/farine_detection
  │         ├── Success → backend = "postgres"
  │         └── Fail    → backend = "sqlite" (fallback)
  └── NO  → backend = "sqlite"
```

### PostgreSQL Connection Pool

```python
self._pg_pool = ThreadedConnectionPool(minconn=1, maxconn=5, ...)
```
- Lazy init: pool created on first `_get_pg_conn()` call
- Each caller: `conn = _get_pg_conn()` ... `_release_pg_conn(conn)` (return to pool)
- Max 5 simultaneous PG connections
- `autocommit=False` → explicit `.commit()` after writes

### SQLite Dual Connections (WAL mode)

```
_sqlite_conn      ← writes (INSERT, UPDATE, DELETE)  ← _sqlite_lock
_sqlite_read_conn ← reads  (SELECT only)             ← _sqlite_read_lock
```
- WAL journal mode allows concurrent readers + one writer
- `_sqlite_read_conn` has `PRAGMA query_only=ON`
- Reads from Flask greenlets never block behind writes from the DBWriter thread

### DBWriter Background Thread

```
Detector thread                    DBWriter thread                    PostgreSQL/SQLite
     │                                  │                                   │
     │  write_queue.put_nowait({        │                                   │
     │    type: "session_update",       │                                   │
     │    session_id, total, ok, nok    │                                   │
     │  })                              │                                   │
     │ ─────────────────────────────→   │                                   │
     │                                  │ event = write_queue.get()         │
     │                                  │ ──────────────────────────────→   │
     │                                  │   UPDATE sessions SET total=...   │
     │                                  │   COMMIT                          │
     │                                  │ ←──────────────────────────────   │
```

- `write_queue` = `queue.Queue(maxsize=10000)` — thread-safe, bounded
- The DBWriter thread loops: `get(timeout=1.0)` → execute → `task_done()`
- Two event types: `session_update` (UPDATE sessions SET totals) and `crossing` (INSERT into defective_packets)
- Frequency: every `SNAPSHOT_EVERY_N_PACKETS` (25) packets, plus every NOK crossing

### Tables

| Table | Purpose |
|-------|---------|
| `sessions` | One row per recording session (id, group_id, shift_id, started_at, ended_at, totals) |
| `defective_packets` | One row per NOK packet (session_id, packet_num, defect_type, crossed_at) |
| `shifts` | Recurring + one-off shift definitions (schedule, camera, checkpoint) |
| `shift_variants` | Per-shift date-range overrides (disable/change times for specific dates) |

---

## 7. HTTP Server & Request Flow

### gevent WSGIServer

```
Browser/Frontend
     │
     │  HTTP request (TCP socket)
     │
     ▼
┌─────────────────────────────┐
│  gevent WSGIServer          │
│  (epoll event loop)         │
│                             │
│  Accepts connection →       │
│  spawns greenlet            │
│                             │
│  Greenlet runs Flask WSGI   │
│  app → route handler        │
│  → returns response         │
│                             │
│  During I/O or sleep():     │
│  greenlet yields to hub,    │
│  other greenlets can run    │
└─────────────────────────────┘
```

- **NOT multithreaded** for request handling — all greenlets share the MainThread
- **Cooperative scheduling** — a greenlet only yields when it does I/O or explicitly sleeps
- **CPU-bound work** in a greenlet blocks ALL other greenlets (this is the key limitation)

### Key API Endpoints

| Endpoint | Method | Purpose | Data source |
|----------|--------|---------|-------------|
| `/video_feed` | GET | MJPEG stream (persistent connection) | `_jpeg_bytes` from Compositor |
| `/api/stats` | GET | Live detection stats | `stats` dict from Detector |
| `/api/pipelines/<id>/stats` | GET | Per-pipeline stats | Same as above, per pipeline |
| `/api/session/start` | POST | Start all pipelines + recording | Guard check → `start_processing()` + `set_stats_recording()` |
| `/api/session/stop` | POST | Stop all pipelines + recording | `set_stats_recording(False)` + `stop_processing()` |
| `/api/stats/toggle` | POST | Toggle recording on/off | `set_stats_recording(toggle)` |
| `/api/stats/sessions` | GET | Session history (merged by group) | `list_grouped_sessions()` from DB |
| `/api/shifts` | GET/POST/PUT/DELETE | Shift CRUD | Direct DB calls |
| `/api/checkpoint` | POST | Switch YOLO model | `switch_checkpoint()` |
| `/api/start` / `/api/stop` | POST | Start/stop pipelines WITHOUT recording | `start_processing()` / `stop_processing()` |
| `/api/prewarm` | POST | Start pipelines for warmup (no stats) | `start_processing()` only |

### MJPEG `/video_feed` — The Persistent Connection

```python
def generate():
    while True:
        jpeg = _view_state()._jpeg_bytes    # read pre-encoded JPEG
        yield multipart_frame(jpeg)
        time.sleep(0.066)                    # ~15fps, yields greenlet
```

- This is a **streaming response** — the HTTP connection stays open indefinitely
- The greenlet runs `while True` but yields to the event loop during `time.sleep(0.066)`
- Other greenlets (API requests) can run during those 66ms windows
- When the browser closes the tab, the TCP connection drops, the greenlet exits

---

## 8. Session Lifecycle

### Manual Session (user clicks "Start" in dashboard)

```
Frontend                    Backend                         DB
   │                           │                             │
   │  POST /api/session/start  │                             │
   │ ────────────────────────→ │                             │
   │                           │ with _session_lock:         │
   │                           │   check guard               │
   │                           │   set guard "manual"        │
   │                           │                             │
   │                           │ for each pipeline:          │
   │                           │   start_processing(source)  │
   │                           │   set_stats_recording(True) │
   │                           │     → db_writer.open_session│
   │                           │ ──────────────────────────→ │ INSERT sessions
   │                           │                             │
   │  { status: "started" }    │                             │
   │ ←──────────────────────── │                             │
   │                           │                             │
   │  GET /api/pipelines/*/stats (every 1500ms)              │
   │ ────────────────────────→ │                             │
   │  { total, ok, nok, ... }  │                             │
   │ ←──────────────────────── │                             │
   │                           │                             │
   │  POST /api/session/stop   │                             │
   │ ────────────────────────→ │                             │
   │                           │ for each pipeline:          │
   │                           │   set_stats_recording(False)│
   │                           │     → db_writer.close_session│
   │                           │ ──────────────────────────→ │ UPDATE sessions SET ended_at
   │                           │   stop_processing()         │
   │                           │                             │
   │                           │ with _session_lock:         │
   │                           │   clear guard               │
   │  { status: "stopped" }    │                             │
   │ ←──────────────────────── │                             │
```

### Scheduled Session (APScheduler fires shift)

```
APScheduler thread              Backend                         DB
   │                               │                             │
   │ _shift_start(shift_id)        │                             │
   │ ───────────────────────────→  │                             │
   │                               │ with _session_lock:         │
   │                               │   check guard (skip if busy)│
   │                               │   check variants (skip if   │
   │                               │     disabled today)          │
   │                               │   set guard "shift"          │
   │                               │                             │
   │                               │ for each pipeline:          │
   │                               │   switch_checkpoint if needed│
   │                               │   start_processing(source)  │
   │                               │   set_stats_recording(True) │
   │                               │ ──────────────────────────→ │ INSERT sessions
   │                               │                             │
   │     (at shift end_time)       │                             │
   │                               │                             │
   │ _shift_stop(shift_id)         │                             │
   │ ───────────────────────────→  │                             │
   │                               │ with _session_lock:         │
   │                               │   verify this shift owns it │
   │                               │   close sessions + stop     │
   │                               │   clear guard               │
```

### Session Guard (`_session_lock`)

Global state protected by `threading.Lock()`:
- `_active_session_source`: `"manual"` | `"shift"` | `None`
- `_active_session_group`: UUID string linking all pipeline sessions
- `_active_session_shift_id`: which shift owns the session (if source="shift")

**Purpose:** Prevents a scheduled shift from starting while a manual session is active, and vice versa.

---

## 9. Frontend Polling (Dashboard)

File: `gmcb-dashboard/src/hooks/useLiveStats.ts`

```
Every 1500ms:
  GET /api/pipelines/pipeline_0/stats
  GET /api/pipelines/pipeline_1/stats

Every 30s:
  GET /api/shifts
  GET /api/one-off-sessions
  GET /api/session/status

Every 60s:
  GET /api/stats/sessions

Persistent:
  GET /video_feed (MJPEG — one long-lived connection)
```

---

## 10. GPU Usage

Both pipelines share 1 GPU (CUDA):

| Model | VRAM | Runs on | Frequency |
|-------|------|---------|-----------|
| YOLO `yolo26m_BB_barcode_date.pt` | ~100MB | Detector p0 | Every frame (30fps) |
| YOLO `yolo26-BB(date).pt` (secondary) | ~100MB | `_secondary_executor` | Every frame (30fps, parallel) |
| YOLO `yolo26m_seg_farine_FV.pt` | ~100MB | Detector p1 | Every 3rd frame (~10fps) |
| EfficientAD (teacher+student+AE) | ~50MB | Detector p1 | Per crop in scanning zone |

CUDA operations release the Python GIL during kernel execution, but Python-side tensor operations (pre/post processing) hold the GIL.

---

## 11. File System Layout

```
demo_detection_realtime/
├── web_server_backend_v2.py    ← Entry point + Flask routes + scheduler
├── tracking_state.py           ← TrackingState class (3 threads per pipeline)
├── tracking_config.py          ← All config: checkpoints, pipelines, thresholds
├── db_writer.py                ← DBWriter class (SQLite + PG, async write queue)
├── db_config.py                ← DB connection params
├── helpers.py                  ← calculate_bbox_metrics, letterbox_image
├── efficientad.py              ← EfficientAD predict() function
├── anomaly_on_video.py         ← get_ad_constants() helper
├── common.py                   ← Shared utilities
├── data/
│   └── tracking_demo.db        ← SQLite database (fallback)
├── liveImages/
│   └── <session_id>/
│       ├── nobarcode/
│       │   └── packet_N.png    ← Proof images for tracking mode NOK
│       ├── nodate/
│       │   └── packet_N.png
│       └── anomalie/
│           └── N/
│               ├── worst_scan.png
│               └── scans.csv   ← Per-scan scores for anomaly mode NOK
├── templates/
│   └── index.html              ← Legacy template (served at /)
├── *.pt / *.pth                ← Model weights (YOLO + EfficientAD)
├── docker-compose.yml          ← PostgreSQL container (port 5434)
└── requirements_web.txt        ← Python dependencies
```

---

## 12. Key Concurrency Mechanisms

| Mechanism | What it protects | Used by |
|-----------|-----------------|---------|
| `_session_lock` (threading.Lock) | `_active_session_source/group/shift_id` globals | `api_session_start`, `api_session_stop`, `api_stats_toggle`, `api_session_reset_guard`, `_shift_start`, `_shift_stop` |
| `_raw_lock` (threading.Lock) | `_raw_frame` latest frame | Reader ↔ Compositor |
| `_det_lock` (threading.Lock) | `_det_frame` frame for detector | Reader ↔ Detector |
| `_overlay_lock` (threading.Lock) | `_overlay` dict | Detector ↔ Compositor |
| `_jpeg_lock` (threading.Lock) | `_jpeg_bytes` encoded frame | Compositor ↔ /video_feed |
| `_stats_lock` (threading.Lock) | `stats` dict | Detector ↔ /api/stats |
| `_perf_lock` (threading.Lock) | `_perf` dict | Detector+Compositor ↔ /api/perf |
| `_raw_history_lock` (threading.Lock) | `_raw_history` deque | Reader ↔ Compositor |
| `_sqlite_lock` (threading.Lock) | Write connection | DBWriter thread ↔ Flask writes |
| `_sqlite_read_lock` (threading.Lock) | Read connection | Flask read greenlets |
| `_det_event` (threading.Event) | "new frame for detector" signal | Reader → Detector |
| `_raw_changed` (threading.Event) | "new raw frame" signal | Reader → Compositor |
| `write_queue` (queue.Queue) | Async DB events | Detector → DBWriter |
| `_pg_pool` (ThreadedConnectionPool) | PG connections | All PG callers |
| `_secondary_executor` (ThreadPoolExecutor) | Secondary model thread | Detector p0 |
| `_proof_executor` (ThreadPoolExecutor) | Proof-image save threads | Detector p0 + p1 |

---

## 13. Known Architectural Constraints

1. **Single process / single GPU** — all inference shares one CUDA context. GPU contention between pipeline_0 (tracking YOLO) and pipeline_1 (segmentation YOLO + EfficientAD) is handled by CUDA's internal scheduling, not by the application.

2. **gevent greenlets ≠ parallelism** — all HTTP handling runs in one OS thread. A CPU-bound greenlet (e.g., building a large JSON response) blocks all other greenlets until it yields. Only I/O and `time.sleep()` yield.

3. **thread=False means no cooperative threading** — the 10+ OS threads (pipeline readers, detectors, compositors, DBWriter, APScheduler) all compete for the Python GIL normally. gevent only makes HTTP request handling cooperative.

4. **`_secondary_executor` and `_proof_executor` are module-level singletons** — shared across all pipelines. The secondary executor has 1 worker (only pipeline_0 uses it). The proof executor has 4 workers (shared by both pipelines).

5. **MJPEG feed serves only one pipeline at a time** — controlled by `active_view_id` global. Switching view pipeline changes which `_jpeg_bytes` the generator reads.

---

## 14. Data Flow Summary (One Detection Cycle)

```
Camera/File → Reader → rotated frame
                        ├──→ _raw_frame (Compositor picks up)
                        └──→ _det_frame (every Nth frame)
                                │
                    Detector picks up _det_frame
                                │
                    ┌───────────┼───────────────┐
                    │           │               │
              Primary YOLO   Secondary YOLO   (tracking mode)
              .track()       (via executor)
                    │           │
                    └─────┬─────┘
                          │
                  Merge results (deduplicate dates)
                          │
                  Per-track processing:
                    • Assign barcode/date to package
                    • Check exit line crossing
                    • Lock decision: OK or NOK
                          │
                  ┌───────┴────────┐
                  │                │
            Update _overlay   If NOK:
            Update stats        • proof image → _proof_executor
                  │             • crossing → write_queue
                  │             If every 25 packets:
                  │               • session_update → write_queue
                  │
         Compositor picks up _overlay + _raw_frame
                  │
         Draw boxes + labels + exit line
         cv2.imencode → _jpeg_bytes
                  │
         /video_feed greenlet reads _jpeg_bytes
         → streams to browser
```
