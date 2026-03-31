# Performance Audit — Server Unreachable During Active Session

**Date:** 2026-03-30  
**Symptom:** Once a session starts (manual or scheduled), the Flask server becomes extremely slow/unreachable. The frontend cannot load live stats, the MJPEG video feed freezes or fails to connect, and the situation only resolves by killing and restarting the backend.

---

## Root Cause Summary

The server uses **Flask's built-in Werkzeug dev server** (`threaded=True`) which spawns one OS thread per request with no limit. Combined with Python's **Global Interpreter Lock (GIL)**, the concurrent threads (video reader, YOLO detector, compositor, MJPEG streaming, DB writer, APScheduler, Flask request handlers) all compete for a single execution lock. During a session, ~10+ threads are active **permanently**, and the frontend's aggressive 400ms polling creates a thread avalanche that starves the server.

---

## Bug #1: MJPEG `/video_feed` Generator Holds a Thread Forever

**Severity: CRITICAL**  
**File:** `web_server_backend_v2.py`, lines 199–220

```python
def generate():
    while True:
        st = _view_state()
        if st is not None:
            with st._jpeg_lock:
                jpeg = st._jpeg_bytes
        else:
            jpeg = None
        frame_bytes = jpeg if jpeg is not None else placeholder_bytes
        yield (b'--frame\r\n' ...)
        time.sleep(0.066)   # ~15 fps — this thread NEVER returns
```

Each browser/tab that connects to `/video_feed` spawns a thread that runs **forever** in a `while True` loop. This thread:
- Holds the GIL during Python execution (lock acquire, bytes concat, `str().encode()`)
- Releases GIL only during `time.sleep(0.066)` — but at 15 fps that's only 66ms gaps
- Never terminates until the browser closes the connection

**Impact:** 1 MJPEG connection = 1 permanently occupied thread. If the frontend opens /video_feed from 2 components (or 2 browser tabs), that's 2 immortal threads competing for GIL.

---

## Bug #2: Frontend Polls 5+ Requests Every 400ms — Thread Avalanche

**Severity: CRITICAL**  
**Files:** `gmcb-dashboard/src/hooks/useLiveStats.ts`, `backendApi.ts`

The frontend fires these requests **every 400ms**:
```
GET /api/pipelines/pipeline_0/stats
GET /api/pipelines/pipeline_1/stats
```

Plus periodic polls:
- Every 30s: `GET /api/shifts` (useShifts)
- Every 30s: `GET /api/one-off-sessions` (useOneOffSessions)
- Every 30s: `GET /api/session/status` (session guard check)
- Every 60s: `GET /api/stats/sessions` (session history)
- Persistent: 1× MJPEG `/video_feed` connection

**At steady state: 5 req/s from stats polling + 1 permanent MJPEG thread.**

When the GIL is contended (during a session), Flask responses slow down from <5ms to 50-200ms. Since the next poll fires before the previous one returns, **threads pile up**:

```
Time 0ms:    poll_1 starts (thread A)
Time 400ms:  poll_2 starts (thread B) — poll_1 still blocked on GIL
Time 800ms:  poll_3 starts (thread C) — both still blocked
...
Time 8000ms: AbortController kills poll_1 — but 20 threads were spawned
```

The frontend has an 8-second timeout (`backendApi.ts`). In worst case: `8000 / 400 × 2 = 40 concurrent threads` waiting for GIL time, plus the persistent MJPEG thread, plus the 3 pipeline threads, plus DBWriter, plus APScheduler = **50+ threads** fighting over one GIL.

**This is why the server becomes unreachable.**

---

## Bug #3: GIL Contention from cv2 Operations

**Severity: HIGH**  
**File:** `tracking_state.py`

OpenCV's C++ functions **do NOT release the Python GIL**:

| Operation | Thread | Frequency | GIL held |
|-----------|--------|-----------|----------|
| `cv2.rotate()` | Reader | Every frame (30 fps) | ~1ms |
| `cv2.imencode('.jpg')` | Compositor | Every frame (~15 fps) | 2-5ms |
| `cv2.rectangle()` / `cv2.putText()` | Compositor | Every frame | ~1ms |
| `cv2.resize()` + `cv2.threshold()` | Detector (anomaly) | Per crop | 1-2ms |

YOLO's `.track()` releases GIL during CUDA kernel execution, but its Python-side pre/post-processing (tensor creation, NMS, result parsing) holds the GIL for 5-15ms per frame.

**Total GIL held per second by pipeline threads: ~100-200ms.** During those windows, ALL Flask request threads are blocked.

---

## Bug #4: Secondary Date Model Doubles GPU + GIL Time

**Severity: HIGH**  
**File:** `tracking_state.py`, detection loop (lines ~1568–1582)

The `barcode_date` checkpoint runs **two sequential YOLO forward passes** per detector frame:

```python
# Primary: YOLO + ByteTrack (~20ms GPU + ~10ms Python)
results = self.model.track(frame, ...)

# Secondary: date detection (~20ms GPU + ~10ms Python)
if self._use_secondary_date and self.secondary_model is not None:
    sec_results = self.secondary_model(frame, ...)
```

This doubles `det_ms` from ~30ms to ~60ms and increases GIL contention proportionally. With `DETECTOR_FRAME_SKIP=1`, the detector runs on **every frame** (30 fps), so the GIL is occupied by YOLO Python code for ~30ms × 30 = ~900ms per second — **90% GIL saturation** from the detector alone.

---

## Bug #5: SQLite `_sqlite_lock` Serializes All DB Access

**Severity: HIGH**  
**File:** `db_writer.py`

All database reads (Flask routes) and writes (background thread) share a single `_sqlite_lock`:

```
Flask thread A: GET /api/shifts → acquire _sqlite_lock → query → release
Flask thread B: GET /api/stats/sessions → BLOCKED waiting for _sqlite_lock
DBWriter thread: session_update → BLOCKED waiting for _sqlite_lock
```

When the detector thread enqueues session updates every 25 packets, the background writer holds `_sqlite_lock` during `UPDATE + COMMIT`. All concurrent Flask DB reads serialize behind this lock.

---

## Bug #6: PostgreSQL Single Connection — No Thread Safety

**Severity: HIGH**  
**File:** `db_writer.py`, lines 608-633

```python
def _get_pg_conn(self):
    for _ in range(3):
        try:
            self._pg_conn = psycopg2.connect(connect_timeout=3, ...)
            return self._pg_conn
        except Exception:
            time.sleep(1.0)  # 3 retries × 4s = 12s blocked
    return None
```

- **Single `self._pg_conn`** shared across all Flask threads — psycopg2 connections are NOT thread-safe
- If PG connection drops, each request blocks for up to **12 seconds** attempting reconnect
- No connection pooling — each reconnect creates a new TCP socket

---

## Bug #7: Session Guard Race Condition

**Severity: MEDIUM**  
**File:** `web_server_backend_v2.py`

The `_session_lock` is only used by scheduler functions (`_shift_start`, `_shift_stop`). Manual session endpoints modify the same globals **without the lock**:

```python
# api_session_start() — NO LOCK
_active_session_source = "manual"
_active_session_group = group_id

# _shift_start() — USES LOCK
with _session_lock:
    if _active_session_source is not None:
        ...
```

If a scheduled shift fires while a manual session start is in progress, both can proceed, creating duplicate sessions.

---

## Bug #8: Proof-Image Threads Spawn Without Limit

**Severity: MEDIUM**  
**File:** `tracking_state.py`, `_save_proof_image_bg()` and `_save_nok_packet_bg()`

Every defective packet spawns a new daemon thread for file I/O:
```python
threading.Thread(target=self._save_proof_image, ..., daemon=True).start()
```

If 100 defective packets cross in quick succession, 100 threads are spawned. Each does `os.makedirs()` + `cv2.imwrite()` which holds the GIL during the PNG encode.

---

## Fix Plan

### Fix 1: Replace Werkzeug with Gunicorn + gevent (Eliminates Bug #1, #2, #6)

Switch from the single-process Werkzeug dev server to **Gunicorn with gevent workers**. Gevent uses greenlets (coroutines) instead of OS threads, so the MJPEG `while True` generator and 400ms polling don't create real OS threads:

```bash
pip install gunicorn gevent
gunicorn -w 1 -k gevent --worker-connections 200 \
         -b 0.0.0.0:5000 web_server_backend_v2:app
```

**Why 1 worker:** The TrackingState objects (model, GPU tensors) live in process memory. Multiple workers would duplicate GPU models. Gevent solves the concurrency problem within a single process via cooperative scheduling.

**Note:** Gevent monkey-patches `time.sleep()` to be cooperative, so the MJPEG generator's `time.sleep(0.066)` yields to other greenlets instead of blocking an OS thread.

### Fix 2: Increase Frontend Polling Interval (Reduces Bug #2)

In `useLiveStats.ts`, change `POLL_MS` from 400 to 1000-1500ms:
```typescript
const POLL_MS = 1000;  // was 400
```

400ms is too aggressive — the human eye cannot distinguish stats updates faster than ~1s. This alone cuts request volume by 60%.

### Fix 3: Run Secondary Date Model in Parallel Thread (Fixes Bug #4)

Use `concurrent.futures.ThreadPoolExecutor` to run the secondary model concurrently with primary result processing:

```python
from concurrent.futures import ThreadPoolExecutor
_secondary_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SecDate")

# In detection loop:
sec_future = None
if self._use_secondary_date and self.secondary_model is not None:
    sec_future = _secondary_executor.submit(
        self.secondary_model, frame, conf=sec_conf, imgsz=CONFIG["imgsz"], verbose=False
    )

# ... process primary results ...

if sec_future is not None:
    sec_results = sec_future.result()[0]
    # merge date detections
```

Since YOLO releases GIL during CUDA execution, both models can run on GPU simultaneously if there's enough VRAM.

### Fix 4: Separate SQLite Read Connection (Fixes Bug #5)

SQLite in WAL mode allows concurrent readers with one writer. Create a dedicated read-only connection for Flask routes:

```python
class DBWriter:
    def __init__(self):
        ...
        self._read_conn = None       # separate connection for reads
        self._read_lock = threading.Lock()  # lighter lock, no write contention

    def _init_sqlite(self):
        ...
        # Read-only connection — WAL mode allows concurrent readers
        self._read_conn = sqlite3.connect(_SQLITE_PATH, check_same_thread=False)
        self._read_conn.row_factory = sqlite3.Row
        self._read_conn.execute("PRAGMA journal_mode=WAL")
```

### Fix 5: Add PostgreSQL Connection Pooling (Fixes Bug #6)

```python
from psycopg2.pool import ThreadedConnectionPool

self._pg_pool = ThreadedConnectionPool(
    minconn=2, maxconn=10,
    host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
    user=DB_USER, password=DB_PASSWORD,
    connect_timeout=3,
)
```

### Fix 6: Use a Thread Pool for Proof Images (Fixes Bug #8)

Replace unbounded `Thread().start()` with a bounded executor:

```python
_proof_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="Proof")

def _save_proof_image_bg(self, ...):
    _proof_executor.submit(self._save_proof_image, ...)
```

### Fix 7: Apply Session Lock Consistently (Fixes Bug #7)

Wrap manual session endpoints with `_session_lock`:

```python
@app.route('/api/session/start', methods=['POST'])
def api_session_start():
    with _session_lock:
        if _active_session_source is not None:
            return jsonify({"error": "session already active"}), 409
        _active_session_source = "manual"
        ...
```

---

## Priority Order

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| **P0** | Fix 1: Gunicorn + gevent | Eliminates thread starvation | Low |
| **P0** | Fix 2: Increase poll interval | Cuts request load 60% | Trivial |
| **P1** | Fix 4: Separate SQLite read conn | Eliminates DB lock contention | Low |
| **P1** | Fix 3: Parallel secondary model | Halves GIL time from detection | Medium |
| **P2** | Fix 5: PG connection pool | Prevents 12s reconnect blocks | Low |
| **P2** | Fix 6: Bounded proof thread pool | Prevents thread explosion | Trivial |
| **P3** | Fix 7: Session lock consistency | Prevents race condition | Trivial |

---

## Thread Count During Active Session (Current State)

| Thread | Lifetime | GIL Impact |
|--------|----------|------------|
| Reader (per pipeline × 2) | Session duration | Low (V4L2 releases GIL) |
| Detector (per pipeline × 2) | Session duration | HIGH (YOLO pre/post + secondary model) |
| Compositor (per pipeline × 2) | Session duration | HIGH (cv2.imencode) |
| MJPEG generator (per browser connection) | Forever | Medium (lock + bytes) |
| DBWriter | App lifetime | Low (short COMMIT bursts) |
| APScheduler (2 threads) | App lifetime | Negligible |
| Flask request handlers (5+/sec) | Per request (~50-200ms) | Medium (JSON + DB) |
| Proof-image savers (up to 100) | Per save (~50ms) | Medium (cv2.imwrite) |
| **TOTAL** | **15-50 threads** | **Massive GIL contention** |
