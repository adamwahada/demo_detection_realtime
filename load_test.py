#!/usr/bin/env python3
"""
Load test for the anomaly detection server.
Simulates full production load: 2 pipelines running with video files,
frontend polling, MJPEG stream, and reports server health every 5s.

Usage:
    # 1. Start the backend first:
    #    cd /home/adam/demo_detection_realtime && python web_server_backend_v2.py

    # 2. In a second terminal, run this test:
    #    python load_test.py

    # 3. Optional: override host/port
    #    python load_test.py --host 127.0.0.1 --port 5000

    # 4. Stop with Ctrl+C — it prints a final report.
"""

import argparse
import sys
import time
import threading
import signal
import requests
from datetime import datetime
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
POLL_INTERVAL_S = 1.5       # mirrors new useLiveStats POLL_MS=1500
SESSION_POLL_S  = 30        # shift/one-off/session-status polls
MJPEG_DURATION_S = 120      # how long to hold the MJPEG connection
TEST_DURATION_S  = 120      # total test time (seconds)

# ── Shared state ─────────────────────────────────────────────────────────────
_lock = threading.Lock()
_results = defaultdict(lambda: {"ok": 0, "err": 0, "total_ms": 0.0, "timeouts": 0})
_stop = threading.Event()

def _record(label, ok, elapsed_ms, timed_out=False):
    with _lock:
        r = _results[label]
        r["ok" if ok else "err"] += 1
        r["total_ms"] += elapsed_ms
        if timed_out:
            r["timeouts"] += 1

def _get(session, url, label, timeout=8):
    t0 = time.time()
    timed_out = False
    try:
        resp = session.get(url, timeout=timeout)
        ok = resp.status_code == 200
        elapsed = (time.time() - t0) * 1000
        _record(label, ok, elapsed)
        return ok, elapsed
    except requests.exceptions.Timeout:
        elapsed = (time.time() - t0) * 1000
        _record(label, False, elapsed, timed_out=True)
        return False, elapsed
    except Exception:
        elapsed = (time.time() - t0) * 1000
        _record(label, False, elapsed)
        return False, elapsed

# ── Worker threads ────────────────────────────────────────────────────────────

def worker_pipeline_stats(base_url, pipeline_id):
    """Polls /api/pipelines/<id>/stats every POLL_INTERVAL_S — mirrors frontend."""
    url = f"{base_url}/api/pipelines/{pipeline_id}/stats"
    label = f"stats/{pipeline_id}"
    s = requests.Session()
    while not _stop.is_set():
        _get(s, url, label)
        _stop.wait(POLL_INTERVAL_S)

def worker_session_status(base_url):
    """Polls /api/session/status every SESSION_POLL_S."""
    s = requests.Session()
    while not _stop.is_set():
        _get(s, f"{base_url}/api/session/status", "session/status")
        _stop.wait(SESSION_POLL_S)

def worker_shifts(base_url):
    """Polls /api/shifts every SESSION_POLL_S."""
    s = requests.Session()
    while not _stop.is_set():
        _get(s, f"{base_url}/api/shifts", "shifts")
        _stop.wait(SESSION_POLL_S)

def worker_one_off(base_url):
    """Polls /api/one-off-sessions every SESSION_POLL_S."""
    s = requests.Session()
    while not _stop.is_set():
        _get(s, f"{base_url}/api/one-off-sessions", "one-off-sessions")
        _stop.wait(SESSION_POLL_S)

def worker_session_history(base_url):
    """Polls /api/stats/sessions every 60s."""
    s = requests.Session()
    while not _stop.is_set():
        _get(s, f"{base_url}/api/stats/sessions", "stats/sessions")
        _stop.wait(60)

def worker_mjpeg(base_url):
    """Holds a persistent MJPEG connection — simulates one browser tab open."""
    url = f"{base_url}/video_feed"
    label = "mjpeg_stream"
    t0 = time.time()
    frames = 0
    try:
        with requests.get(url, stream=True, timeout=10) as resp:
            if resp.status_code != 200:
                _record(label, False, 0)
                return
            for chunk in resp.iter_content(chunk_size=8192):
                if _stop.is_set():
                    break
                if b'--frame' in chunk:
                    frames += 1
                if time.time() - t0 > MJPEG_DURATION_S:
                    break
    except Exception as e:
        pass
    elapsed = (time.time() - t0) * 1000
    _record(label, frames > 0, elapsed)
    print(f"  [MJPEG] Received {frames} frames over {elapsed/1000:.1f}s")

def worker_start_session(base_url):
    """POSTs /api/session/start once, then /api/session/stop at the end."""
    time.sleep(2)  # let other workers stabilize first
    s = requests.Session()
    t0 = time.time()
    try:
        resp = s.post(f"{base_url}/api/session/start", json={}, timeout=10)
        ok = resp.status_code in (200, 409)  # 409 = already active, still fine
        elapsed = (time.time() - t0) * 1000
        _record("session/start", ok, elapsed)
        status = resp.status_code
        body = resp.json() if resp.headers.get("content-type","").startswith("application/json") else {}
        print(f"  [SESSION] Start → HTTP {status}  group={body.get('group_id','')[:8]}…  {elapsed:.0f}ms")
    except Exception as e:
        print(f"  [SESSION] Start failed: {e}")

def worker_stop_session(base_url, delay):
    """Stops the session after `delay` seconds."""
    time.sleep(delay)
    if _stop.is_set():
        return
    s = requests.Session()
    t0 = time.time()
    try:
        resp = s.post(f"{base_url}/api/session/stop", json={}, timeout=10)
        elapsed = (time.time() - t0) * 1000
        _record("session/stop", resp.status_code == 200, elapsed)
        print(f"  [SESSION] Stop  → HTTP {resp.status_code}  {elapsed:.0f}ms")
    except Exception as e:
        print(f"  [SESSION] Stop failed: {e}")

# ── Health reporter ───────────────────────────────────────────────────────────

def reporter(interval=5):
    while not _stop.is_set():
        _stop.wait(interval)
        if _stop.is_set():
            break
        now = datetime.now().strftime("%H:%M:%S")
        with _lock:
            snap = {k: dict(v) for k, v in _results.items()}
        lines = [f"\n{'─'*62}  {now}"]
        lines.append(f"  {'Endpoint':<28} {'OK':>5} {'ERR':>5} {'TIMEOUT':>8} {'avg ms':>8}")
        lines.append(f"  {'─'*28} {'─'*5} {'─'*5} {'─'*8} {'─'*8}")
        for label, r in sorted(snap.items()):
            total = r["ok"] + r["err"]
            avg = r["total_ms"] / total if total else 0
            flag = " ⚠" if r["err"] > 0 or r["timeouts"] > 0 else ""
            lines.append(f"  {label:<28} {r['ok']:>5} {r['err']:>5} {r['timeouts']:>8} {avg:>7.0f}ms{flag}")
        print("\n".join(lines))

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--duration", type=int, default=TEST_DURATION_S,
                        help="Test duration in seconds (default 120)")
    parser.add_argument("--no-session", action="store_true",
                        help="Skip starting a recording session (just test idle load)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    duration = args.duration

    # Quick connectivity check
    print(f"\n{'='*62}")
    print(f"  Load test → {base_url}")
    print(f"  Duration : {duration}s")
    print(f"  Pipelines: pipeline_0 (tracking), pipeline_1 (anomaly)")
    print(f"{'='*62}\n")

    try:
        r = requests.get(f"{base_url}/api/pipelines", timeout=5)
        pls = r.json().get("pipelines", [])
        for p in pls:
            run = "running" if p["is_running"] else "stopped"
            rec = " + recording" if p["stats_active"] else ""
            print(f"  [{p['id']}] {p['label']} — {run}{rec}")
    except Exception as e:
        print(f"  ✗ Cannot reach server at {base_url}: {e}")
        print("  Make sure the backend is running first.")
        sys.exit(1)
    print()

    threads = []

    def t(fn, *a):
        th = threading.Thread(target=fn, args=a, daemon=True)
        th.start()
        threads.append(th)

    # Simulate exactly what the frontend does during a shift
    t(worker_pipeline_stats, base_url, "pipeline_0")   # stats poll pipeline 0
    t(worker_pipeline_stats, base_url, "pipeline_1")   # stats poll pipeline 1
    t(worker_session_status, base_url)                  # session guard check
    t(worker_shifts,         base_url)                  # shift list
    t(worker_one_off,        base_url)                  # one-off list
    t(worker_session_history, base_url)                 # history
    t(worker_mjpeg,          base_url)                  # MJPEG stream (1 browser tab)
    t(reporter)                                         # live health table

    if not args.no_session:
        # Start a session after 2s, stop it 10s before the end
        t(worker_start_session, base_url)
        t(worker_stop_session,  base_url, duration - 10)

    signal.signal(signal.SIGINT, lambda *_: _stop.set())

    print(f"  Test running for {duration}s — Ctrl+C to stop early\n")
    _stop.wait(duration)
    _stop.set()

    # Final report
    time.sleep(1)
    with _lock:
        snap = {k: dict(v) for k, v in _results.items()}

    print(f"\n{'='*62}")
    print(f"  FINAL REPORT  ({duration}s test)")
    print(f"{'='*62}")
    print(f"  {'Endpoint':<28} {'OK':>5} {'ERR':>5} {'TIMEOUT':>8} {'avg ms':>8}  {'p_err':>6}")
    print(f"  {'─'*28} {'─'*5} {'─'*5} {'─'*8} {'─'*8}  {'─'*6}")

    all_ok = True
    for label, r in sorted(snap.items()):
        total = r["ok"] + r["err"]
        avg = r["total_ms"] / total if total else 0
        pct_err = r["err"] / total * 100 if total else 0
        flag = " ← PROBLEM" if pct_err > 5 or r["timeouts"] > 2 else ""
        if flag:
            all_ok = False
        print(f"  {label:<28} {r['ok']:>5} {r['err']:>5} {r['timeouts']:>8} {avg:>7.0f}ms  {pct_err:>5.1f}%{flag}")

    print(f"\n{'='*62}")
    if all_ok:
        print("  ✓ Server held up — all endpoints responsive under load")
    else:
        print("  ✗ Issues detected — review lines marked ← PROBLEM")
        print("    Common causes:")
        print("    • avg ms > 500 on stats/* → GIL still contended (check monkey.patch_all)")
        print("    • timeouts > 0 on mjpeg   → MJPEG generator blocking greenlets")
        print("    • ERR on session/start     → session guard collision")
    print(f"{'='*62}\n")

if __name__ == "__main__":
    main()