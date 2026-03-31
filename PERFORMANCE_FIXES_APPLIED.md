# Performance Fixes Applied

**Date:** 2026-03-30  
**Reference:** `PERFORMANCE_AUDIT.md` (same date)

All 7 fixes from the audit have been implemented. Below is what changed in each file.

---

## Fix 1 — gevent WSGIServer (Bugs #1, #2, #3)

**File:** `web_server_backend_v2.py`

- Added `from gevent import monkey; monkey.patch_all()` at the very top of the file (before all other imports) so that `time.sleep`, `threading.Lock`, `socket`, etc. are all monkey-patched to cooperative greenlets.
- Replaced the bottom `app.run(host=..., threaded=True)` block with `gevent.pywsgi.WSGIServer((host, port), app).serve_forever()`.
- The MJPEG `while True` generator in `/video_feed` now yields to other greenlets during `time.sleep(0.066)` instead of holding an OS thread hostage. This eliminates the permanent thread-per-viewer problem.
- All Flask request handlers now run as lightweight greenlets — no more OS-thread avalanche from concurrent polling.

**File:** `requirements_web.txt`

- Added `gevent` to the dependency list.

---

## Fix 2 — Frontend Poll Interval 400ms → 1500ms (Bug #2)

**File:** `gmcb-dashboard/src/hooks/useLiveStats.ts`

- Changed `POLL_MS` from `400` to `1500`.
- Reduces peak request volume from ~5 req/s to ~1.3 req/s, cutting thread/greenlet pressure roughly 4×.

---

## Fix 3 — SQLite Separate Read Connection (Bug #5)

**File:** `db_writer.py`

- Added `_sqlite_read_conn` (a second `sqlite3.connect` with `check_same_thread=False`) and its own `_sqlite_read_lock`.
- Added helper method `_sqlite_read(sql, params)` that executes a SELECT on the read connection.
- All read-only methods that serve Flask routes (`get_session`, `list_grouped_sessions`, `list_crossings`, `get_shifts`, `get_one_off_sessions`, `get_variants`, etc.) now use `_sqlite_read()` instead of acquiring `_sqlite_lock` on the write connection.
- Write operations (`open_session`, `close_session`, `_update_session_live`, `_write_crossing`, shift/variant CRUD) continue to use the original `_sqlite_conn` + `_sqlite_lock`.
- Result: Flask read queries no longer block behind DB writes, and vice versa.

---

## Fix 4 — Secondary Date Model Runs in Parallel (Bug #4)

**File:** `tracking_state.py`

- Added `from concurrent.futures import ThreadPoolExecutor` and module-level `_secondary_executor = ThreadPoolExecutor(max_workers=1)`.
- In `_detection_loop`, the secondary date model is now **submitted** to the executor **before** extracting primary results (tracks, barcodes, dates):
  ```python
  sec_future = _secondary_executor.submit(self.secondary_model, frame, ...)
  ```
- After primary result extraction is done, the secondary result is **collected**:
  ```python
  sec_results = sec_future.result(timeout=2.0)[0]
  ```
- This overlaps secondary YOLO inference with the CPU-bound primary result parsing, reducing effective `det_ms` by ~15-25ms per frame when the secondary model is active.

---

## Fix 5 — PostgreSQL Connection Pool (Bug #6)

**File:** `db_writer.py`

- Replaced single `self._pg_conn` with `psycopg2.pool.ThreadedConnectionPool` (`_pg_pool`, minconn=1, maxconn=5).
- Added `_get_pg_conn()` that calls `self._pg_pool.getconn()`.
- Added `_release_pg_conn(conn)` that calls `self._pg_pool.putconn(conn)`.
- Every PG method now follows the pattern:
  ```python
  conn = self._get_pg_conn()
  try:
      # use conn
  except Exception:
      # log error
  finally:
      self._release_pg_conn(conn)
  ```
- All 4 legacy `self._pg_conn = None` error handlers (in `_run`, `insert_one_off_session`, `update_one_off_session`, `delete_one_off_session`) were replaced with proper `try/finally: _release_pg_conn(conn)`.
- Eliminates thread-safety issues from shared connection and removes 12-second reconnect blocking.

---

## Fix 6 — Bounded Proof-Image Thread Pool (Bug #8)

**File:** `tracking_state.py`

- Added module-level `_proof_executor = ThreadPoolExecutor(max_workers=4)`.
- `_save_proof_image_bg()` now uses `_proof_executor.submit(...)` instead of `threading.Thread(...).start()`.
- `_save_nok_packet_bg()` now uses `_proof_executor.submit(...)` instead of `threading.Thread(...).start()`.
- Maximum 4 concurrent proof-image saves at any time. Excess work queues inside the executor instead of spawning unbounded OS threads.

---

## Fix 7 — Session Lock Consistency (Bug #7)

**File:** `web_server_backend_v2.py`

- `api_session_start()`: guard check (`_active_session_source is not None`) and state mutation are now wrapped in `with _session_lock:`. Pipeline starts remain outside the lock.
- `api_session_stop()`: guard clearing wrapped in `with _session_lock:`.
- `api_stats_toggle()`: guard mutations wrapped in `with _session_lock:`.
- `api_session_reset_guard()`: guard clearing wrapped in `with _session_lock:`.
- The scheduler functions (`_shift_start`, `_shift_stop`) already used `_session_lock`. Now manual and scheduled paths are fully synchronized — no more race between a manual start and a scheduled shift.

---

## Deployment Checklist

1. `pip install gevent` in the runtime environment
2. Restart the backend (`python web_server_backend_v2.py`)
3. Rebuild/redeploy the frontend (poll interval change in `useLiveStats.ts`)
