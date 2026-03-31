from gevent import monkey
monkey.patch_all(thread=False, queue=False)   # keep real OS threads + queues for PyTorch/CUDA & ThreadPoolExecutor

import atexit
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response, render_template, send_file
from flask_cors import CORS
from ultralytics import YOLO

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.base import JobLookupError

from tracking_config import (
    MODEL_PATH, PACKAGE_CLASS_NAME, BARCODE_CLASS_NAME,
    CONFIG, SERVER_HOST, SERVER_PORT,
    CHECKPOINTS, CAMERAS,
    DEFAULT_CHECKPOINT_ID, DEFAULT_CAMERA_ID,
    get_checkpoint, get_camera,
    DEVICE,
    PIPELINES, DEFAULT_VIEW_PIPELINE,
)
from tracking_state import TrackingState

try:
    from db_writer import DBWriter
    _DB_AVAILABLE = True
except Exception:
    DBWriter = None
    _DB_AVAILABLE = False

# ==========================
# FLASK APP
# ==========================
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app, resources={r"/api/*": {"origins": "*"},
                     r"/video_feed": {"origins": "*"}})

# ==========================
# GLOBAL STATE — multi-pipeline
# ==========================
# Shared DB writer (one instance, used by all pipelines)
db_writer = DBWriter() if _DB_AVAILABLE else None

# Pipeline dict: pipeline_id → TrackingState
pipelines: dict = {}

# Which pipeline serves /video_feed right now
active_view_id: str = DEFAULT_VIEW_PIPELINE

# Per-pipeline checkpoint tracking
pipeline_checkpoint_ids: dict = {}   # pipeline_id → checkpoint_id

# Active-session guard: prevents scheduler vs manual collisions
# _active_session_source: "shift" | "manual" | None
# _active_session_group: the group_id of the currently running session
# _active_session_shift_id: if source is "shift", which shift owns the session
_active_session_source = None
_active_session_group = None
_active_session_shift_id = None
_session_lock = threading.Lock()   # guards _active_session_* reads/writes
_TUNIS_TZ = ZoneInfo("Africa/Tunis")


def _view_state() -> TrackingState:
    """Return the TrackingState currently selected for live viewing."""
    return pipelines.get(active_view_id)


def _all_states():
    """Yield all (pipeline_id, TrackingState) tuples."""
    return pipelines.items()


# ==========================
# MODEL INIT (per pipeline)
# ==========================
def init_pipeline(pipe_cfg):
    """Create a TrackingState for one pipeline config and load its model."""
    pid = pipe_cfg["id"]
    cp_id = pipe_cfg["checkpoint_id"]
    checkpoint = get_checkpoint(cp_id)
    if checkpoint is None:
        raise ValueError(f"Unknown checkpoint id: {cp_id} for pipeline {pid}")

    state = TrackingState(pipeline_id=pid, db_writer=db_writer)

    print(f"[{pid}] Loading model: {checkpoint['label']} ({checkpoint['path']})...")
    state.model = YOLO(checkpoint["path"])
    state.model.to(DEVICE)
    names = state.model.names

    pkg_cls = checkpoint.get("package_class")
    bar_cls = checkpoint.get("barcode_class")
    date_cls = checkpoint.get("date_class")
    state.package_id = next((k for k, v in names.items() if v == pkg_cls), None) if pkg_cls else None
    state.barcode_id = next((k for k, v in names.items() if v == bar_cls), None) if bar_cls else None
    state.date_id = next((k for k, v in names.items() if v == date_cls), None) if date_cls else None
    state.mode = checkpoint.get("mode", "tracking")
    state.current_checkpoint = checkpoint
    print(f"[{pid}] Model loaded on {DEVICE}. mode={state.mode} "
          f"package={state.package_id} barcode={state.barcode_id} date={state.date_id}")

    # Load secondary date model when configured on tracking checkpoints
    sec_path = checkpoint.get("secondary_date_model_path")
    sec_cls = checkpoint.get("secondary_date_class")
    if sec_path and state.mode == "tracking":
        try:
            state.secondary_model = YOLO(sec_path)
            state.secondary_model.to(DEVICE)
            sec_names = state.secondary_model.names
            state._secondary_date_id = next(
                (k for k, v in sec_names.items() if v == sec_cls), None
            ) if sec_cls else None
            state._use_secondary_date = state._secondary_date_id is not None
            print(f"[{pid}] Secondary date model loaded: id={state._secondary_date_id}")
        except Exception as e:
            state.secondary_model = None
            state._secondary_date_id = None
            state._use_secondary_date = False
            print(f"[{pid}] Secondary date model load failed (non-fatal): {e}")
    else:
        state.secondary_model = None
        state._secondary_date_id = None
        state._use_secondary_date = False

    # If this checkpoint is the anomaly detector, attempt to register/load AD models
    if state.mode == "anomaly":
        try:
            state._load_ad_models(checkpoint, DEVICE)
        except Exception as e:
            print(f"[{pid}][AD] _load_ad_models failed: {e}")

    print(f"[{pid}] Warming up YOLO (dummy inference)...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        state.model(dummy, imgsz=CONFIG["imgsz"], verbose=False)
        import torch
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        print(f"[{pid}] YOLO warmup complete.")
    except Exception as e:
        print(f"[{pid}] YOLO warmup failed (non-fatal): {e}")

    pipelines[pid] = state
    pipeline_checkpoint_ids[pid] = cp_id
    return state


def init_all_pipelines():
    """Initialize all configured pipelines. Called once at startup."""
    for pipe_cfg in PIPELINES:
        init_pipeline(pipe_cfg)
    print(f"[INIT] {len(pipelines)} pipeline(s) initialized")


# ==========================
# WEB ROUTES
# ==========================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG stream — serves pre-encoded JPEG bytes from the active view pipeline."""
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for stream...", (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    _, buf = cv2.imencode('.jpg', placeholder)
    placeholder_bytes = buf.tobytes()

    def generate():
        while True:
            st = _view_state()
            if st is not None:
                with st._jpeg_lock:
                    jpeg = st._jpeg_bytes
            else:
                jpeg = None
            frame_bytes = jpeg if jpeg is not None else placeholder_bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                   + frame_bytes + b'\r\n')
            time.sleep(0.066)

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/api/start', methods=['POST'])
def api_start():
    """Start all pipelines.
    Optional body: { "sources": { "pipeline_0": "video.mp4", "pipeline_1": 1 } }
    If sources dict is provided, each pipeline uses its override; otherwise uses
    the camera_source from PIPELINES config.
    """
    data = request.get_json(silent=True) or {}
    source_overrides = data.get("sources", {})
    results = {}
    for pipe_cfg in PIPELINES:
        pid = pipe_cfg["id"]
        st = pipelines.get(pid)
        if st is None:
            continue
        source = source_overrides.get(pid, pipe_cfg["camera_source"])
        # Coerce numeric string to int for camera index
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        results[pid] = st.start_processing(source)
    return jsonify({"status": "started", "pipelines": results})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop all pipelines."""
    results = {}
    for pid, st in _all_states():
        results[pid] = st.stop_processing()
    return jsonify({"status": "stopped", "pipelines": results})


@app.route('/api/prewarm', methods=['POST'])
def api_prewarm():
    """Start all pipelines (camera + inference) WITHOUT starting stats recording.
    Used to warm up models before the actual shift start so the first packets
    are not missed due to cold-start latency.
    Optional body: { "sources": { "pipeline_0": "video.mp4", ... } }
    """
    data = request.get_json(silent=True) or {}
    source_overrides = data.get("sources", {})
    results = {}
    for pipe_cfg in PIPELINES:
        pid = pipe_cfg["id"]
        st = pipelines.get(pid)
        if st is None:
            continue
        if st.is_running:
            results[pid] = {"status": "already_running"}
            continue
        source = source_overrides.get(pid, pipe_cfg["camera_source"])
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        results[pid] = st.start_processing(source)
        print(f"[PREWARM][{pid}] Pipeline started (no stats recording)")
    return jsonify({"status": "prewarmed", "pipelines": results})


@app.route('/api/prewarm/status', methods=['GET'])
def api_prewarm_status():
    """Return per-pipeline warm status: is_running (camera active) vs stats_active (recording)."""
    result = {}
    for pid, st in _all_states():
        result[pid] = {
            "is_running": st.is_running,
            "stats_active": getattr(st, '_stats_active', False),
        }
    any_warm = any(v["is_running"] and not v["stats_active"] for v in result.values())
    any_recording = any(v["stats_active"] for v in result.values())
    return jsonify({
        "pipelines": result,
        "is_prewarmed": any_warm,
        "is_recording": any_recording,
    })


@app.route('/api/pause', methods=['POST'])
def api_pause():
    results = {}
    for pid, st in _all_states():
        results[pid] = st.pause_processing()
    return jsonify({"status": "paused", "pipelines": results})


@app.route('/api/resume', methods=['POST'])
def api_resume():
    results = {}
    for pid, st in _all_states():
        results[pid] = st.resume_processing()
    return jsonify({"status": "resumed", "pipelines": results})


@app.route('/api/stats')
def api_stats():
    """Return stats for the active view pipeline."""
    state = _view_state()
    if state is None:
        return jsonify({"error": "no active pipeline"}), 404
    pid = active_view_id
    cp_id = pipeline_checkpoint_ids.get(pid, "")
    with state._stats_lock:
        s = dict(state.stats)
    with state._perf_lock:
        perf = dict(state._perf)
    s["pipeline_id"]      = pid
    s["checkpoint_id"]    = cp_id
    s["checkpoint_label"] = (state.current_checkpoint or {}).get("label", "")
    s["checkpoint_mode"]  = state.mode
    s["camera_id"]        = ""
    s["exit_line_enabled"] = state._exit_line_enabled
    s["exit_line_vertical"] = state._exit_line_vertical
    s["exit_line_inverted"] = state._exit_line_inverted
    s["exit_line_pct"] = state._exit_line_pct
    s["rotation_deg"] = (state._rotation_steps % 4) * 90
    s["perf"] = perf
    s["is_paused"] = getattr(state, '_is_paused', False)
    s["stats_active"] = getattr(state, '_stats_active', False)
    s["session_id"] = getattr(state, '_db_session_id', None)
    s["db_available"] = db_writer is not None
    s["db_backend"] = db_writer.backend if db_writer is not None else None
    s["nok_no_barcode"] = getattr(state, '_nok_no_barcode', 0)
    s["nok_no_date"] = getattr(state, '_nok_no_date', 0)
    return jsonify(s)


@app.route('/api/perf')
def api_perf():
    state = _view_state()
    if state is None:
        return jsonify({"error": "no active pipeline"}), 404
    pid = active_view_id
    with state._stats_lock:
        stats = dict(state.stats)
    with state._perf_lock:
        perf = dict(state._perf)

    return jsonify({
        "pipeline_id": pid,
        "checkpoint_id": pipeline_checkpoint_ids.get(pid, ""),
        "checkpoint_mode": state.mode,
        "is_running": state.is_running,
        "frame_count": state.frame_count,
        "video_fps": stats.get("video_fps", 0),
        "det_fps": stats.get("det_fps", 0),
        "inference_ms": stats.get("inference_ms", 0),
        "perf": perf,
    })


@app.route('/api/stats/status')
def api_stats_status():
    state = _view_state()
    if state is None:
        return jsonify({"error": "no active pipeline"}), 404
    total = state.total_packets
    ok_count = state.output_fifo.count("OK")
    nok_count = state.output_fifo.count("NOK")
    return jsonify({
        "pipeline_id": active_view_id,
        "stats_active": getattr(state, '_stats_active', False),
        "session_id": getattr(state, '_db_session_id', None),
        "db_available": db_writer is not None,
        "db_backend": db_writer.backend if db_writer is not None else None,
        "total": total,
        "ok_count": ok_count,
        "nok_count": nok_count,
        "nok_no_barcode": getattr(state, '_nok_no_barcode', 0),
        "nok_no_date": getattr(state, '_nok_no_date', 0),
        "nok_rate_pct": round(nok_count / total * 100, 2) if total > 0 else 0.0,
    })


@app.route('/api/stats/toggle', methods=['POST'])
def api_stats_toggle():
    """Toggle stats recording on ALL pipelines simultaneously.
    All pipelines that start together share the same group_id so their
    sessions can be merged in the dashboard as a single logical run.
    """
    global _active_session_source, _active_session_group, _active_session_shift_id

    request.get_json(force=True, silent=True)
    any_active = any(getattr(st, '_stats_active', False) for _, st in _all_states())
    new_active = not any_active

    # One shared group_id for this toggle-on event — links all pipeline sessions
    group_id = str(uuid.uuid4()) if new_active else ""

    with _session_lock:
        if new_active:
            _active_session_source = "manual"
            _active_session_group = group_id
            _active_session_shift_id = None
        else:
            _active_session_source = None
            _active_session_group = None
            _active_session_shift_id = None

    results = {}
    for pid, st in _all_states():
        results[pid] = st.set_stats_recording(new_active, group_id=group_id)
    return jsonify({
        "stats_active": new_active,
        "group_id": group_id,
        "db_available": db_writer is not None,
        "db_backend": db_writer.backend if db_writer is not None else None,
        "pipelines": results,
    })


@app.route('/api/session/start', methods=['POST'])
def api_session_start():
    """Unified start: start all pipelines + open DB sessions in one call.
    Accepts optional body: { "shift_id": "..." } to link to a planned shift.
    """
    global _active_session_source, _active_session_group, _active_session_shift_id

    with _session_lock:
        if _active_session_source is not None:
            return jsonify({
                "error": "session already active",
                "source": _active_session_source,
                "group_id": _active_session_group,
            }), 409

        data = request.get_json(force=True, silent=True) or {}
        shift_id = (data.get("shift_id") or "").strip()

        group_id = str(uuid.uuid4())
        _active_session_source = "manual"
        _active_session_group = group_id
        _active_session_shift_id = shift_id or None

    source_overrides = data.get("sources", {})

    pipeline_results = {}
    for pipe_cfg in PIPELINES:
        pid = pipe_cfg["id"]
        st = pipelines.get(pid)
        if st is None:
            continue
        source = source_overrides.get(pid, pipe_cfg["camera_source"])
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        if not st.is_running:
            st.start_processing(source)
        st.set_stats_recording(True, group_id=group_id, shift_id=shift_id)
        pipeline_results[pid] = "started"

    return jsonify({
        "status": "started",
        "group_id": group_id,
        "shift_id": shift_id,
        "pipelines": pipeline_results,
    })


@app.route('/api/session/stop', methods=['POST'])
def api_session_stop():
    """Unified stop: close DB sessions + stop all pipelines in one call."""
    global _active_session_source, _active_session_group, _active_session_shift_id

    pipeline_results = {}
    for pid, st in _all_states():
        if getattr(st, '_stats_active', False):
            st.set_stats_recording(False)
        if st.is_running:
            st.stop_processing()
        pipeline_results[pid] = "stopped"

    with _session_lock:
        prev_group = _active_session_group
        _active_session_source = None
        _active_session_group = None
        _active_session_shift_id = None

    return jsonify({
        "status": "stopped",
        "group_id": prev_group or "",
        "pipelines": pipeline_results,
    })


@app.route('/api/session/status')
def api_session_status():
    """Return current session guard status."""
    any_running = any(st.is_running for _, st in _all_states())
    any_recording = any(getattr(st, '_stats_active', False) for _, st in _all_states())
    guard_stale = _active_session_source is not None and not any_running and not any_recording
    return jsonify({
        "active": _active_session_source is not None,
        "source": _active_session_source,
        "group_id": _active_session_group,
        "shift_id": _active_session_shift_id,
        "any_running": any_running,
        "any_recording": any_recording,
        "guard_stale": guard_stale,
    })


@app.route('/api/session/reset-guard', methods=['POST'])
def api_session_reset_guard():
    """Force-clear the session guard. Use when guard is stuck (e.g. after a crash).
    Does NOT stop any pipeline — only resets the guard state."""
    global _active_session_source, _active_session_group, _active_session_shift_id
    with _session_lock:
        prev = _active_session_source
        _active_session_source = None
        _active_session_group = None
        _active_session_shift_id = None
    print(f"[SESSION] Guard manually reset (was: {prev})")
    return jsonify({"reset": True, "previous_source": prev})


@app.route('/api/stats/sessions')
def api_stats_sessions():
    """Returns sessions grouped by group_id (merged across pipelines)."""
    limit = request.args.get('limit', default=50, type=int)
    if db_writer is None:
        return jsonify({"sessions": []})
    sessions = db_writer.list_grouped_sessions(limit=max(1, min(limit, 500)))
    return jsonify({"sessions": sessions})


@app.route('/api/stats/sessions/raw')
def api_stats_sessions_raw():
    """Returns raw individual sessions (one per pipeline)."""
    limit = request.args.get('limit', default=50, type=int)
    if db_writer is None:
        return jsonify({"sessions": []})
    sessions = db_writer.list_sessions(limit=max(1, min(limit, 500)))
    return jsonify({"sessions": sessions})


@app.route('/api/stats/session/<session_id>')
def api_stats_session(session_id):
    if db_writer is None:
        return jsonify({}), 404
    data = db_writer.get_session_kpis(session_id)
    if not data:
        return jsonify({}), 404
    return jsonify(data)


@app.route('/api/stats/session/<session_id>/crossings')
def api_stats_session_crossings(session_id):
    """Crossings for a group_id (merged) or a raw session_id.
    First tries group lookup; if no rows found falls back to single-session lookup.
    """
    if db_writer is None:
        return jsonify({"crossings": []})
    limit = request.args.get('limit', default=5000, type=int)
    capped = max(1, min(limit, 10000))
    # Try group lookup first
    rows = db_writer.list_crossings_for_group(session_id, limit=capped)
    if not rows:
        # Fallback: raw session id (legacy or single-pipeline)
        rows = db_writer.list_crossings(session_id, limit=capped)
    return jsonify({"crossings": rows})


@app.route('/api/proof/<session_id>/<defect_type>/<int:packet_num>')
def api_proof_image(session_id, defect_type, packet_num):
    """Serve a proof image for a defective packet."""
    if defect_type not in ("nobarcode", "nodate", "anomaly"):
        return jsonify({"error": "invalid defect_type"}), 400

    def _related_session_ids(target_session_id):
        candidates = [target_session_id]
        if db_writer is None or not target_session_id:
            return candidates

        try:
            raw = db_writer.get_session_kpis(target_session_id)
        except Exception:
            raw = {}

        group_id = raw.get("group_id") if isinstance(raw, dict) else ""
        if group_id:
            candidates.append(group_id)

        try:
            for row in db_writer.list_sessions(limit=2000):
                row_id = row.get("id")
                row_group = row.get("group_id") or ""
                if row_id == target_session_id and row_group:
                    candidates.append(row_group)
                if row_group == target_session_id and row_id:
                    candidates.append(row_id)
        except Exception:
            pass

        seen = set()
        ordered = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
        return ordered

    def _candidate_paths(base_dir):
        if defect_type == "anomaly":
            return [
                base_dir / "anomalie" / str(packet_num) / "worst_scan.png",
                base_dir / "anomalie" / str(packet_num) / "worst_scan.jpg",
                base_dir / "anomaly" / str(packet_num) / "worst_scan.png",
                base_dir / "anomaly" / str(packet_num) / "worst_scan.jpg",
            ]
        return [
            base_dir / defect_type / f"packet_{packet_num}.png",
            base_dir / defect_type / f"packet_{packet_num}.jpg",
            base_dir / defect_type / f"packet_{packet_num}.jpeg",
        ]

    live_images_root = Path(__file__).parent / "liveImages"
    img_path = None
    for candidate_session_id in _related_session_ids(session_id):
        candidate_base = live_images_root / candidate_session_id
        for candidate_path in _candidate_paths(candidate_base):
            if candidate_path.is_file():
                img_path = candidate_path
                break
        if img_path is not None:
            break

    if img_path is None:
        return jsonify({"error": "image not found"}), 404

    resp = send_file(img_path, mimetype="image/png")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        data = request.get_json()
        for key, value in data.items():
            if key in CONFIG:
                CONFIG[key] = value
        return jsonify({"status": "updated", "config": CONFIG})
    return jsonify(CONFIG)


@app.route('/api/fifo')
def api_fifo():
    state = _view_state()
    if state is None:
        return jsonify({"fifo": [], "total_packets": 0})
    return jsonify({
        "fifo": state.output_fifo,
        "total_packets": state.total_packets
    })


@app.route('/api/exit_line', methods=['POST'])
def api_exit_line():
    """Toggle exit line on/off on the active view pipeline."""
    state = _view_state()
    if state is None:
        return jsonify({"error": "no active pipeline"}), 404
    state._exit_line_enabled = not state._exit_line_enabled
    return jsonify({"enabled": state._exit_line_enabled})


@app.route('/api/exit_line_orientation', methods=['POST'])
def api_exit_line_orientation():
    state = _view_state()
    if state is None:
        return jsonify({"error": "no active pipeline"}), 404
    state._exit_line_vertical = not state._exit_line_vertical
    state._recompute_exit_line_y()
    orientation = "vertical" if state._exit_line_vertical else "horizontal"
    print(f'[EXIT LINE] Orientation set to {orientation}')
    return jsonify({"vertical": state._exit_line_vertical, "orientation": orientation})


@app.route('/api/exit_line_invert', methods=['POST'])
def api_exit_line_invert():
    state = _view_state()
    if state is None:
        return jsonify({"error": "no active pipeline"}), 404
    state._exit_line_inverted = not state._exit_line_inverted
    state._recompute_exit_line_y()
    print(f'[EXIT LINE] Direction inverted={state._exit_line_inverted}')
    return jsonify({"inverted": state._exit_line_inverted})


@app.route('/api/exit_line_position', methods=['POST'])
def api_exit_line_position():
    state = _view_state()
    if state is None:
        return jsonify({"error": "no active pipeline"}), 404
    data = request.get_json() or {}
    pct = data.get('position', 85)
    try:
        pct = max(5, min(95, int(pct)))
    except (TypeError, ValueError):
        return jsonify({'error': 'position must be an integer 5-95'}), 400

    CONFIG['exit_line_ratio'] = round(1.0 - pct / 100.0, 4)
    state._exit_line_pct = pct

    if state._frame_height > 0:
        state._recompute_exit_line_y()
        print(f'[EXIT LINE] Position set to {pct}% (y/x={state._exit_line_y}, rot={state._rotation_steps * 90}° CCW)')
    else:
        print(f'[EXIT LINE] Position queued at {pct}% (will apply on next Start)')
    return jsonify({'position_pct': pct, 'exit_line_y': state._exit_line_y})


@app.route('/api/rotate', methods=['POST'])
def api_rotate():
    state = _view_state()
    if state is None:
        return jsonify({"error": "no active pipeline"}), 404
    deg = state.cycle_rotation_ccw()
    return jsonify({"rotation_deg": deg})


@app.route('/api/checkpoints')
def api_checkpoints():
    """List available checkpoints."""
    return jsonify({
        "checkpoints": CHECKPOINTS,
    })


@app.route('/api/cameras')
def api_cameras():
    """List available cameras."""
    return jsonify({
        "cameras": CAMERAS,
    })


@app.route('/api/switch', methods=['POST'])
def api_switch():
    """
    Switch checkpoint and/or camera on the active view pipeline.
    Body: { "checkpoint_id": "date", "camera_id": "cam1" }
    """
    state = _view_state()
    if state is None:
        return jsonify({"error": "no active pipeline"}), 404
    pid = active_view_id

    data = request.get_json() or {}
    new_cp_id  = data.get("checkpoint_id")
    new_cam_id = data.get("camera_id")
    custom_src = data.get("custom_source")

    # Resolve new source
    new_source = None
    if custom_src:
        new_source = custom_src
    elif new_cam_id:
        cam = get_camera(new_cam_id)
        if cam is None:
            return jsonify({"error": f"Unknown camera id: {new_cam_id}"}), 400
        new_source = cam["source"]

    cur_cp_id = pipeline_checkpoint_ids.get(pid, "")

    # If only camera changed (not checkpoint), just restart with new source
    if new_cp_id is None or new_cp_id == cur_cp_id:
        if new_source and state.is_running:
            state.stop_processing()
            state.start_processing(new_source)
        return jsonify({
            "status": "camera_switched",
            "pipeline_id": pid,
            "checkpoint_id": cur_cp_id,
            "source": new_source,
        })

    # Checkpoint switch: unload old, load new, restart if was running
    checkpoint = get_checkpoint(new_cp_id)
    if checkpoint is None:
        return jsonify({"error": f"Unknown checkpoint id: {new_cp_id}"}), 400

    was_running = state.is_running
    prev_source = state.video_source

    result = state.switch_checkpoint(checkpoint)
    pipeline_checkpoint_ids[pid] = new_cp_id

    target_source = new_source or prev_source
    if was_running and target_source:
        state.start_processing(target_source)

    result["source"] = target_source
    result["pipeline_id"] = pid
    return jsonify(result)


# ==========================
# PIPELINES API
# ==========================

@app.route('/api/pipelines')
def api_pipelines():
    """List all pipelines with their current state."""
    result = []
    for pipe_cfg in PIPELINES:
        pid = pipe_cfg["id"]
        st = pipelines.get(pid)
        cp_id = pipeline_checkpoint_ids.get(pid, "")
        entry = {
            "id": pid,
            "label": pipe_cfg["label"],
            "camera_source": pipe_cfg["camera_source"],
            "checkpoint_id": cp_id,
            "is_running": st.is_running if st else False,
            "is_paused": getattr(st, '_is_paused', False) if st else False,
            "stats_active": getattr(st, '_stats_active', False) if st else False,
            "session_id": getattr(st, '_db_session_id', None) if st else None,
            "total_packets": st.total_packets if st else 0,
            "is_active_view": pid == active_view_id,
        }
        result.append(entry)
    return jsonify({"pipelines": result, "active_view_id": active_view_id})


@app.route('/api/pipelines/<pipeline_id>/view', methods=['POST'])
def api_pipeline_view(pipeline_id):
    """Switch which pipeline serves /video_feed."""
    global active_view_id
    if pipeline_id not in pipelines:
        return jsonify({"error": f"Unknown pipeline id: {pipeline_id}"}), 404
    active_view_id = pipeline_id
    print(f"[VIEW] Switched active view to {pipeline_id}")
    return jsonify({"active_view_id": active_view_id})


@app.route('/api/pipelines/<pipeline_id>/stats')
def api_pipeline_stats(pipeline_id):
    """Get stats for a specific pipeline (not just the viewed one)."""
    st = pipelines.get(pipeline_id)
    if st is None:
        return jsonify({"error": f"Unknown pipeline id: {pipeline_id}"}), 404
    cp_id = pipeline_checkpoint_ids.get(pipeline_id, "")
    with st._stats_lock:
        s = dict(st.stats)
    with st._perf_lock:
        perf = dict(st._perf)
    s["pipeline_id"] = pipeline_id
    s["checkpoint_id"] = cp_id
    s["checkpoint_label"] = (st.current_checkpoint or {}).get("label", "")
    s["checkpoint_mode"] = st.mode
    s["is_running"] = st.is_running
    s["stats_active"] = getattr(st, '_stats_active', False)
    s["session_id"] = getattr(st, '_db_session_id', None)
    s["total_packets"] = st.total_packets
    s["nok_no_barcode"] = getattr(st, '_nok_no_barcode', 0)
    s["nok_no_date"] = getattr(st, '_nok_no_date', 0)
    s["nok_anomaly"] = getattr(st, '_nok_anomaly', 0)
    s["fifo_queue"] = list(st.output_fifo[-20:]) if hasattr(st, 'output_fifo') else []
    s["perf"] = perf
    return jsonify(s)


@app.route('/api/pipelines/<pipeline_id>/start', methods=['POST'])
def api_pipeline_start(pipeline_id):
    """Start one pipeline with a custom source and/or checkpoint.

    Body (all optional):
      {
        "source":        "path/to/video.mp4" | "0" | 1,   ← camera index or file/RTSP
        "checkpoint_id": "barcode_date"                    ← switch checkpoint before starting
      }
    If source is omitted, uses the pipeline's configured camera_source.
    If checkpoint_id is omitted, uses the currently loaded checkpoint.
    """
    st = pipelines.get(pipeline_id)
    if st is None:
        return jsonify({"error": f"Unknown pipeline id: {pipeline_id}"}), 404

    data = request.get_json(silent=True) or {}
    source = data.get("source")
    new_cp_id = data.get("checkpoint_id")

    # Resolve source — fall back to pipeline config
    if source is None:
        pipe_cfg = next((p for p in PIPELINES if p["id"] == pipeline_id), {})
        source = pipe_cfg.get("camera_source", "0")
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # Switch checkpoint if requested
    if new_cp_id and new_cp_id != pipeline_checkpoint_ids.get(pipeline_id):
        cp = get_checkpoint(new_cp_id)
        if cp is None:
            return jsonify({"error": f"Unknown checkpoint_id: {new_cp_id}"}), 400
        if st.is_running:
            st.stop_processing()
        result = st.switch_checkpoint(cp)
        pipeline_checkpoint_ids[pipeline_id] = new_cp_id
        print(f"[{pipeline_id}] Checkpoint switched to {new_cp_id}: {result.get('status')}")

    if st.is_running:
        return jsonify({"error": "already running", "pipeline_id": pipeline_id}), 409

    res = st.start_processing(source)
    res["pipeline_id"] = pipeline_id
    res["checkpoint_id"] = pipeline_checkpoint_ids.get(pipeline_id, "")
    return jsonify(res)


@app.route('/api/pipelines/<pipeline_id>/stop', methods=['POST'])
def api_pipeline_stop(pipeline_id):
    """Stop one pipeline independently."""
    st = pipelines.get(pipeline_id)
    if st is None:
        return jsonify({"error": f"Unknown pipeline id: {pipeline_id}"}), 404
    res = st.stop_processing()
    res["pipeline_id"] = pipeline_id
    return jsonify(res)


@app.route('/api/pipelines/<pipeline_id>/switch', methods=['POST'])
def api_pipeline_switch(pipeline_id):
    """Switch checkpoint (and optionally source) on one pipeline.

    Body: { "checkpoint_id": "anomaly", "source": "video.mp4" }
    Stops the pipeline if running, loads the new model, restarts if was running.
    """
    st = pipelines.get(pipeline_id)
    if st is None:
        return jsonify({"error": f"Unknown pipeline id: {pipeline_id}"}), 404

    data = request.get_json(silent=True) or {}
    new_cp_id = data.get("checkpoint_id")
    new_source = data.get("source")

    if new_cp_id is None:
        return jsonify({"error": "checkpoint_id is required"}), 400
    cp = get_checkpoint(new_cp_id)
    if cp is None:
        return jsonify({"error": f"Unknown checkpoint_id: {new_cp_id}"}), 400

    if isinstance(new_source, str) and new_source.isdigit():
        new_source = int(new_source)

    prev_source = st.video_source
    was_running = st.is_running

    result = st.switch_checkpoint(cp)
    pipeline_checkpoint_ids[pipeline_id] = new_cp_id

    target_source = new_source or prev_source
    if was_running and target_source:
        st.start_processing(target_source)

    result["pipeline_id"] = pipeline_id
    result["source"] = target_source
    return jsonify(result)


# ==========================
# SHIFTS CRUD
# ==========================
_HH_MM_RE = re.compile(r'^([01]\d|2[0-3]):[0-5]\d$')
_VALID_DAYS = {'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'}
_ISO_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _validate_time(value):
    return isinstance(value, str) and _HH_MM_RE.match(value)


def _validate_days(value):
    if not isinstance(value, list) or len(value) == 0:
        return False
    return all(isinstance(d, str) and d.lower() in _VALID_DAYS for d in value)


def _validate_date(value):
    return isinstance(value, str) and _ISO_DATE_RE.match(value)


def _time_order_ok(start, end):
    """Return True if start < end (HH:MM strings). Midnight-cross shifts not supported."""
    return start < end


def _time_to_minutes(value):
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def _one_off_start_already_passed(date_iso, start_time):
    now_tn = datetime.now(_TUNIS_TZ)
    today_tn = now_tn.date().isoformat()
    if date_iso < today_tn:
        return True
    if date_iso > today_tn:
        return False
    return _time_to_minutes(start_time) < (now_tn.hour * 60 + now_tn.minute)


def _date_order_ok(start, end):
    """Return True if start <= end (ISO date strings)."""
    return start <= end


@app.route('/api/shifts', methods=['GET'])
def api_shifts_list():
    if db_writer is None:
        return jsonify({"shifts": []})
    shifts = db_writer.get_all_shifts()
    return jsonify({"shifts": shifts})


@app.route('/api/shifts', methods=['POST'])
def api_shifts_create():
    if db_writer is None:
        return jsonify({"error": "database not available"}), 503
    data = request.get_json()
    if not data:
        return jsonify({"error": "missing JSON body"}), 400

    label = (data.get("label") or "").strip()
    if not label:
        return jsonify({"error": "label is required"}), 400

    start_time = data.get("start_time", "")
    end_time = data.get("end_time", "")
    if not _validate_time(start_time) or not _validate_time(end_time):
        return jsonify({"error": "start_time and end_time must be HH:MM 24h format"}), 400
    if not _time_order_ok(start_time, end_time):
        return jsonify({"error": "L'heure de début doit être avant l'heure de fin"}), 400

    days_of_week = data.get("days_of_week", [])
    if not _validate_days(days_of_week):
        return jsonify({"error": "days_of_week must be a non-empty array of mon-sun"}), 400

    start_date = data.get("start_date", "")
    end_date = data.get("end_date", "")
    if start_date and not _validate_date(start_date):
        return jsonify({"error": "start_date must be YYYY-MM-DD"}), 400
    if end_date and not _validate_date(end_date):
        return jsonify({"error": "end_date must be YYYY-MM-DD"}), 400
    if start_date and end_date and not _date_order_ok(start_date, end_date):
        return jsonify({"error": "La date de début doit être avant la date de fin"}), 400

    camera_source = str(data.get("camera_source", "0"))
    checkpoint_id = data.get("checkpoint_id", "tracking")
    if get_checkpoint(checkpoint_id) is None:
        return jsonify({"error": f"unknown checkpoint_id: {checkpoint_id}"}), 400

    # Duplicate check: same start+end times with any shared weekday
    existing = db_writer.get_all_shifts()
    new_days = set(d.lower() for d in days_of_week)
    for s in existing:
        if s.get("type", "recurring") != "recurring":
            continue
        if s["start_time"] == start_time and s["end_time"] == end_time:
            try:
                ex_days = set(json.loads(s["days_of_week"]))
            except Exception:
                ex_days = set()
            shared = new_days & ex_days
            if shared:
                return jsonify({"error": f"Un shift identique existe déjà : {s['label']} ({start_time}–{end_time})"}), 409

    shift = {
        "id": str(uuid.uuid4()),
        "label": label,
        "start_time": start_time,
        "end_time": end_time,
        "start_date": start_date or None,
        "end_date": end_date or None,
        "days_of_week": json.dumps([d.lower() for d in days_of_week]),
        "camera_source": camera_source,
        "checkpoint_id": checkpoint_id,
        "active": 1,
        "created_at": datetime.now(_TUNIS_TZ).replace(tzinfo=None).strftime('%Y-%m-%dT%H:%M:%S'),
    }
    ok = db_writer.insert_shift(shift)
    if not ok:
        return jsonify({"error": "insert failed"}), 500

    _reschedule_shift(shift["id"])
    return jsonify({"shift": shift}), 201


@app.route('/api/shifts/<shift_id>', methods=['PUT'])
def api_shifts_update(shift_id):
    if db_writer is None:
        return jsonify({"error": "database not available"}), 503
    data = request.get_json()
    if not data:
        return jsonify({"error": "missing JSON body"}), 400

    fields = {}
    if "label" in data:
        label = (data["label"] or "").strip()
        if not label:
            return jsonify({"error": "label cannot be empty"}), 400
        fields["label"] = label
    if "start_time" in data:
        if not _validate_time(data["start_time"]):
            return jsonify({"error": "start_time must be HH:MM"}), 400
        fields["start_time"] = data["start_time"]
    if "end_time" in data:
        if not _validate_time(data["end_time"]):
            return jsonify({"error": "end_time must be HH:MM"}), 400
        fields["end_time"] = data["end_time"]
    # Cross-check order after both are resolved
    resolved_start = fields.get("start_time") or (db_writer.get_shift(shift_id) or {}).get("start_time", "")
    resolved_end = fields.get("end_time") or (db_writer.get_shift(shift_id) or {}).get("end_time", "")
    if resolved_start and resolved_end and not _time_order_ok(resolved_start, resolved_end):
        return jsonify({"error": "L'heure de début doit être avant l'heure de fin"}), 400
    if "days_of_week" in data:
        if not _validate_days(data["days_of_week"]):
            return jsonify({"error": "days_of_week must be a non-empty array of mon-sun"}), 400
        fields["days_of_week"] = json.dumps([d.lower() for d in data["days_of_week"]])
    if "start_date" in data:
        if data["start_date"] and not _validate_date(data["start_date"]):
            return jsonify({"error": "start_date must be YYYY-MM-DD"}), 400
        fields["start_date"] = data["start_date"] or None
    if "end_date" in data:
        if data["end_date"] and not _validate_date(data["end_date"]):
            return jsonify({"error": "end_date must be YYYY-MM-DD"}), 400
        fields["end_date"] = data["end_date"] or None
    # Cross-check date order if both provided
    resolved_sd = fields.get("start_date") or (db_writer.get_shift(shift_id) or {}).get("start_date", "")
    resolved_ed = fields.get("end_date") or (db_writer.get_shift(shift_id) or {}).get("end_date", "")
    if resolved_sd and resolved_ed and not _date_order_ok(resolved_sd, resolved_ed):
        return jsonify({"error": "La date de début doit être avant la date de fin"}), 400
    if "camera_source" in data:
        fields["camera_source"] = str(data["camera_source"])
    if "checkpoint_id" in data:
        if get_checkpoint(data["checkpoint_id"]) is None:
            return jsonify({"error": f"unknown checkpoint_id: {data['checkpoint_id']}"}), 400
        fields["checkpoint_id"] = data["checkpoint_id"]

    if not fields:
        return jsonify({"error": "no valid fields to update"}), 400

    ok = db_writer.update_shift(shift_id, fields)
    if not ok:
        return jsonify({"error": "update failed"}), 500

    _reschedule_shift(shift_id)
    updated = db_writer.get_shift(shift_id)
    return jsonify({"shift": updated})


@app.route('/api/shifts/<shift_id>', methods=['DELETE'])
def api_shifts_delete(shift_id):
    if db_writer is None:
        return jsonify({"error": "database not available"}), 503
    _remove_shift_jobs(shift_id)
    ok = db_writer.delete_shift(shift_id)
    if not ok:
        return jsonify({"error": "delete failed"}), 500
    return jsonify({"deleted": True})


@app.route('/api/shifts/<shift_id>/toggle', methods=['POST'])
def api_shifts_toggle(shift_id):
    if db_writer is None:
        return jsonify({"error": "database not available"}), 503
    new_active = db_writer.toggle_shift(shift_id)
    if new_active is None:
        return jsonify({"error": "toggle failed"}), 500
    _reschedule_shift(shift_id)
    return jsonify({"id": shift_id, "active": new_active})


@app.route('/api/shifts/<shift_id>/variants', methods=['POST'])
def api_variants_create(shift_id):
    if db_writer is None:
        return jsonify({"error": "database not available"}), 503
    parent = db_writer.get_shift(shift_id)
    if parent is None:
        return jsonify({"error": "shift not found"}), 404
    if parent.get("type") == "one_off":
        return jsonify({"error": "Les shifts ponctuels ne peuvent pas avoir de personnalisations"}), 400
    data = request.get_json()
    if not data:
        return jsonify({"error": "missing JSON body"}), 400

    kind = data.get("kind", "")
    if kind not in ("timing", "availability"):
        return jsonify({"error": "kind must be 'timing' or 'availability'"}), 400

    start_date = data.get("start_date", "")
    end_date = data.get("end_date", "")
    days_of_week = data.get("days_of_week", [])
    if not start_date or not end_date or not days_of_week:
        return jsonify({"error": "start_date, end_date, days_of_week are required"}), 400
    if not _validate_date(start_date) or not _validate_date(end_date):
        return jsonify({"error": "start_date and end_date must be YYYY-MM-DD"}), 400
    if not _date_order_ok(start_date, end_date):
        return jsonify({"error": "La date de début doit être avant la date de fin"}), 400
    if kind == "timing":
        st = data.get("start_time")
        et = data.get("end_time")
        if st and et and not _time_order_ok(st, et):
            return jsonify({"error": "L'heure de début doit être avant l'heure de fin"}), 400

    variant = {
        "id": str(uuid.uuid4()),
        "shift_id": shift_id,
        "kind": kind,
        "active": data.get("active"),
        "start_time": data.get("start_time"),
        "end_time": data.get("end_time"),
        "start_date": start_date,
        "end_date": end_date,
        "days_of_week": json.dumps([d.lower() for d in days_of_week]),
        "created_at": datetime.now(_TUNIS_TZ).replace(tzinfo=None).strftime('%Y-%m-%dT%H:%M:%S'),
    }
    result = db_writer.insert_variant(variant)
    if result is None:
        return jsonify({"error": "insert failed"}), 500
    return jsonify({"variant": result}), 201


@app.route('/api/shifts/<shift_id>/variants/<variant_id>', methods=['PUT'])
def api_variants_update(shift_id, variant_id):
    if db_writer is None:
        return jsonify({"error": "database not available"}), 503
    data = request.get_json()
    if not data:
        return jsonify({"error": "missing JSON body"}), 400

    fields = {}
    if "kind" in data:
        if data["kind"] not in ("timing", "availability"):
            return jsonify({"error": "kind must be 'timing' or 'availability'"}), 400
        fields["kind"] = data["kind"]
    for f in ("active", "start_time", "end_time", "start_date", "end_date"):
        if f in data:
            fields[f] = data[f]
    if "days_of_week" in data:
        fields["days_of_week"] = json.dumps([d.lower() for d in data["days_of_week"]])
    # Date/time order cross-checks
    if "start_date" in fields and "end_date" in fields:
        if not _date_order_ok(fields["start_date"], fields["end_date"]):
            return jsonify({"error": "La date de début doit être avant la date de fin"}), 400
    if "start_time" in fields and "end_time" in fields:
        if not _time_order_ok(fields["start_time"], fields["end_time"]):
            return jsonify({"error": "L'heure de début doit être avant l'heure de fin"}), 400

    if not fields:
        return jsonify({"error": "no valid fields to update"}), 400

    ok = db_writer.update_variant(variant_id, fields)
    if not ok:
        return jsonify({"error": "update failed"}), 500
    return jsonify({"updated": True})


@app.route('/api/shifts/<shift_id>/variants/<variant_id>', methods=['DELETE'])
def api_variants_delete(shift_id, variant_id):
    if db_writer is None:
        return jsonify({"error": "database not available"}), 503
    ok = db_writer.delete_variant(variant_id)
    if not ok:
        return jsonify({"error": "delete failed"}), 500
    return jsonify({"deleted": True})


# ==========================
# ONE-OFF SESSIONS
# ==========================

@app.route('/api/one-off-sessions', methods=['GET'])
def api_one_off_list():
    if db_writer is None:
        return jsonify({"sessions": []})
    sessions = db_writer.get_all_one_off_sessions()
    return jsonify({"sessions": sessions})


@app.route('/api/one-off-sessions', methods=['POST'])
def api_one_off_create():
    if db_writer is None:
        return jsonify({"error": "database not available"}), 503
    data = request.get_json(force=True) or {}
    label = (data.get("label") or "").strip()
    date = (data.get("date") or "").strip()
    start_time = (data.get("start_time") or "").strip()
    end_time = (data.get("end_time") or "").strip()
    if not label or not date or not start_time or not end_time:
        return jsonify({"error": "label, date, start_time, end_time are required"}), 400
    if not _time_order_ok(start_time, end_time):
        return jsonify({"error": "L'heure de début doit être avant l'heure de fin"}), 400
    if _one_off_start_already_passed(date, start_time):
        return jsonify({"error": "Impossible de créer un shift ponctuel si son heure de début est déjà passée"}), 400
    import uuid, datetime as _dt
    session = {
        "id": str(uuid.uuid4()),
        "label": label,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "camera_source": data.get("camera_source", "0"),
        "checkpoint_id": data.get("checkpoint_id", "tracking"),
        "created_at": _dt.datetime.now(_TUNIS_TZ).replace(tzinfo=None).isoformat(),
    }
    result = db_writer.insert_one_off_session(session)
    if result is None:
        return jsonify({"error": "insert failed"}), 500
    # Schedule the one-off in APScheduler
    _remove_shift_jobs(session["id"])
    _schedule_shift({
        "id": session["id"], "label": label, "type": "one_off",
        "start_time": start_time, "end_time": end_time,
        "session_date": date, "active": 1,
    })
    return jsonify({"session": result}), 201


@app.route('/api/one-off-sessions/<session_id>', methods=['PUT'])
def api_one_off_update(session_id):
    if db_writer is None:
        return jsonify({"error": "database not available"}), 503
    data = request.get_json(force=True) or {}
    start_time = (data.get("start_time") or "").strip() or None
    end_time = (data.get("end_time") or "").strip() or None
    if start_time and end_time and not _time_order_ok(start_time, end_time):
        return jsonify({"error": "L'heure de début doit être avant l'heure de fin"}), 400
    current = db_writer.get_shift(session_id) or {}
    resolved_date = current.get("session_date", "")
    resolved_start = start_time or current.get("start_time", "")
    fields = {}
    if start_time:
        fields["start_time"] = start_time
    if end_time:
        fields["end_time"] = end_time
    if not fields:
        return jsonify({"error": "start_time or end_time required"}), 400
    if resolved_date and resolved_start and _one_off_start_already_passed(resolved_date, resolved_start):
        return jsonify({"error": "Impossible de modifier un shift ponctuel dont l'heure de début est déjà passée"}), 400
    ok = db_writer.update_one_off_session(session_id, fields)
    if not ok:
        return jsonify({"error": "update failed"}), 500
    # Reschedule in APScheduler
    _reschedule_shift(session_id)
    return jsonify({"updated": True})


@app.route('/api/one-off-sessions/<session_id>', methods=['DELETE'])
def api_one_off_delete(session_id):
    if db_writer is None:
        return jsonify({"error": "database not available"}), 503
    ok = db_writer.delete_one_off_session(session_id)
    if not ok:
        return jsonify({"error": "delete failed"}), 500
    _remove_shift_jobs(session_id)
    return jsonify({"deleted": True})


# ==========================
# SCHEDULER
# ==========================
scheduler = BackgroundScheduler(
    daemon=True,
    timezone="Africa/Tunis",
)


def _check_shift_variants(shift_id, today_str):
    """Check shift variants for today's date.
    Returns (should_run: bool, time_override: dict|None).
    time_override has 'start_time' / 'end_time' if a timing variant applies.
    """
    variants = db_writer.get_variants_for_shift(shift_id) if db_writer else []
    time_override = None
    for v in variants:
        start_d = v.get("start_date", "")
        end_d = v.get("end_date", "")
        if not (start_d <= today_str <= end_d):
            continue
        # Check day_of_week constraint (if any)
        dow_raw = v.get("days_of_week", "[]")
        if isinstance(dow_raw, str):
            dow_raw = json.loads(dow_raw)
        if dow_raw:
            from datetime import date as _date
            today_dow = _date.fromisoformat(today_str).strftime("%a").lower()[:3]
            if today_dow not in [d.lower()[:3] for d in dow_raw]:
                continue
        kind = v.get("kind", "")
        if kind == "availability" and not v.get("active"):
            return False, None  # shift disabled today
        if kind == "timing":
            time_override = {
                "start_time": v.get("start_time"),
                "end_time": v.get("end_time"),
            }
    return True, time_override


def _shift_prewarm(shift_id):
    """Called by APScheduler 2 minutes before a shift's start_time.
    Starts all pipelines (camera + inference) WITHOUT stats recording so
    models are warm and ready when the shift begins."""
    if db_writer is None:
        return
    shift = db_writer.get_shift(shift_id)
    if not shift or not shift.get("active"):
        return

    label = shift.get("label", shift_id)
    today_str = datetime.now(_TUNIS_TZ).strftime("%Y-%m-%d")
    should_run, _time_ov = _check_shift_variants(shift_id, today_str)
    if not should_run:
        print(f"[PREWARM] Shift '{label}' skipped — disabled by variant for {today_str}")
        return

    print(f"[PREWARM] Shift '{label}' — pre-warming pipelines (2 min before start)")
    for pipe_cfg in PIPELINES:
        pid = pipe_cfg["id"]
        st = pipelines.get(pid)
        if st is None or st.is_running:
            continue
        cam_src = pipe_cfg["camera_source"]
        cp_id = pipe_cfg["checkpoint_id"]
        cur_cp = pipeline_checkpoint_ids.get(pid, "")
        if cp_id != cur_cp:
            cp = get_checkpoint(cp_id)
            if cp is not None:
                st.switch_checkpoint(cp)
                pipeline_checkpoint_ids[pid] = cp_id
        st.start_processing(cam_src)
        print(f"[PREWARM][{pid}] Pipeline started on {cam_src} (no stats recording)")


def _shift_start(shift_id):
    """Called by APScheduler when a shift's start_time fires.
    Starts ALL pipelines with their configured camera+checkpoint."""
    global _active_session_source, _active_session_group, _active_session_shift_id

    if db_writer is None:
        return
    shift = db_writer.get_shift(shift_id)
    if not shift or not shift.get("active"):
        return

    label = shift.get("label", shift_id)

    with _session_lock:
        # Mutual-exclusion guard — skip if something is already actively recording.
        # But ignore a stale guard: if no pipeline is actually running/recording,
        # the guard is orphaned (e.g. from a previous manual session that was stopped
        # via stop-pipeline without going through toggleRecording) — clear it and proceed.
        if _active_session_source is not None:
            any_live = any(
                st.is_running or getattr(st, '_stats_active', False)
                for _, st in _all_states()
            )
            if any_live:
                print(f"[SCHEDULER] Shift '{label}' skipped — pipelines already active "
                      f"(source={_active_session_source}, group={(_active_session_group or '')[:8]})")
                return
            # Stale guard — clear it so this shift can start normally
            print(f"[SCHEDULER] Shift '{label}' — clearing stale guard "
                  f"(source={_active_session_source}, no pipelines running)")
            _active_session_source = None
            _active_session_group = None
            _active_session_shift_id = None

        # Check variants — skip if disabled today
        today_str = datetime.now(_TUNIS_TZ).strftime("%Y-%m-%d")
        should_run, _time_ov = _check_shift_variants(shift_id, today_str)
        if not should_run:
            print(f"[SCHEDULER] Shift '{label}' skipped — disabled by variant for {today_str}")
            return

        print(f"[SCHEDULER] Shift '{label}' starting — activating all pipelines")

        # Shared group_id so all pipeline sessions merge in the dashboard
        group_id = str(uuid.uuid4())

        # Set guard before starting pipelines (still inside lock)
        _active_session_source = "shift"
        _active_session_group = group_id
        _active_session_shift_id = shift_id

    # Pipelines start outside lock (can take time)
    for pipe_cfg in PIPELINES:
        pid = pipe_cfg["id"]
        st = pipelines.get(pid)
        if st is None:
            continue

        cam_src = pipe_cfg["camera_source"]
        cp_id = pipe_cfg["checkpoint_id"]

        # Switch checkpoint if this pipeline's current checkpoint differs
        cur_cp = pipeline_checkpoint_ids.get(pid, "")
        if cp_id != cur_cp:
            cp = get_checkpoint(cp_id)
            if cp is not None:
                result = st.switch_checkpoint(cp)
                pipeline_checkpoint_ids[pid] = cp_id
                print(f"[SCHEDULER][{pid}] Checkpoint switched to {cp_id}: {result.get('status')}")

        # Start processing
        if not st.is_running:
            st.start_processing(cam_src)
            print(f"[SCHEDULER][{pid}] Started on camera {cam_src}")

        # Start stats recording (opens DB session)
        if not getattr(st, '_stats_active', False):
            st.set_stats_recording(True, group_id=group_id, shift_id=shift_id)
            print(f"[SCHEDULER][{pid}] Stats recording started (group {group_id[:8]}…)")

    print(f"[SCHEDULER] Shift '{label}' started automatically — all pipelines active")


def _shift_stop(shift_id):
    """Called by APScheduler when a shift's end_time fires.
    Stops ALL pipelines."""
    global _active_session_source, _active_session_group, _active_session_shift_id

    if db_writer is None:
        return
    shift = db_writer.get_shift(shift_id)
    label = shift.get("label", shift_id) if shift else shift_id

    with _session_lock:
        # Stop only if this shift owns the session, or the session is already gone.
        # If a *different* shift owns it, leave it alone.
        if _active_session_source == "shift" and _active_session_shift_id != shift_id:
            print(f"[SCHEDULER] Shift '{label}' stop skipped — current session owned by different shift")
            return
        # For manual sessions: stop pipelines but reset the guard so the next
        # scheduled shift can fire cleanly.
        if _active_session_source == "manual":
            print(f"[SCHEDULER] Shift '{label}' end-of-window — manual session was active, resetting guard")
            _active_session_source = None
            _active_session_group = None
            _active_session_shift_id = None
            # Don't force-stop the manual session; leave pipelines as they are.
            return

    print(f"[SCHEDULER] Shift '{label}' stopping — deactivating all pipelines")

    for pid, st in _all_states():
        # Stop stats recording (closes DB session with totals)
        if getattr(st, '_stats_active', False):
            st.set_stats_recording(False)
            print(f"[SCHEDULER][{pid}] Stats recording stopped")
        # Stop processing (camera + threads)
        if st.is_running:
            st.stop_processing()
            print(f"[SCHEDULER][{pid}] Stopped")

    # Clear guard
    with _session_lock:
        _active_session_source = None
        _active_session_group = None
        _active_session_shift_id = None

    print(f"[SCHEDULER] Shift '{label}' stopped automatically — all pipelines inactive")


def _remove_shift_jobs(shift_id):
    """Remove start/stop jobs for a shift (if they exist)."""
    for prefix in ("start_", "stop_", "prewarm_"):
        try:
            scheduler.remove_job(f"{prefix}{shift_id}")
        except JobLookupError:
            pass


def _schedule_shift(shift):
    """Add start + stop jobs for one active shift.
    Recurring shifts use CronTrigger; one-off shifts use DateTrigger.
    """
    shift_id = shift["id"]
    shift_type = shift.get("type", "recurring")
    s_hour, s_min = shift["start_time"].split(":")
    e_hour, e_min = shift["end_time"].split(":")

    if shift_type == "one_off":
        # One-off: fire exactly once on session_date at start_time / end_time
        session_date = shift.get("session_date", "")
        if not session_date:
            print(f"[SCHEDULER] One-off shift '{shift.get('label', shift_id)}' has no session_date — skipping")
            return
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Africa/Tunis")
        try:
            start_dt = _dt.strptime(f"{session_date} {shift['start_time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
            end_dt = _dt.strptime(f"{session_date} {shift['end_time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
        except ValueError as e:
            print(f"[SCHEDULER] One-off shift date parse error: {e}")
            return
        # Don't schedule if the date has already passed
        if end_dt < _dt.now(tz):
            print(f"[SCHEDULER] One-off shift '{shift.get('label', shift_id)}' already past — skipping")
            return
        prewarm_dt = start_dt - timedelta(minutes=2)
        if prewarm_dt > _dt.now(tz):
            scheduler.add_job(
                _shift_prewarm, DateTrigger(run_date=prewarm_dt),
                id=f"prewarm_{shift_id}", args=[shift_id], replace_existing=True,
            )
        scheduler.add_job(
            _shift_start, DateTrigger(run_date=start_dt),
            id=f"start_{shift_id}", args=[shift_id], replace_existing=True,
        )
        scheduler.add_job(
            _shift_stop, DateTrigger(run_date=end_dt),
            id=f"stop_{shift_id}", args=[shift_id], replace_existing=True,
        )
        print(f"[SCHEDULER] Scheduled one-off '{shift.get('label', shift_id)}' "
              f"on {session_date} {shift['start_time']}-{shift['end_time']}")
    else:
        # Recurring: CronTrigger on days_of_week
        days_raw = shift["days_of_week"]
        if isinstance(days_raw, str):
            days_raw = json.loads(days_raw)
        day_str = ",".join(d.lower() for d in days_raw)
        # Optional date bounds for the recurring trigger
        cron_start = shift.get("start_date") or None
        cron_end = shift.get("end_date") or None
        # Prewarm job: fire 2 minutes before start
        pre_total = int(s_hour) * 60 + int(s_min) - 2
        if pre_total < 0:
            pre_total += 24 * 60
        pre_hour, pre_min = divmod(pre_total, 60)
        scheduler.add_job(
            _shift_prewarm,
            CronTrigger(day_of_week=day_str, hour=pre_hour, minute=pre_min,
                        timezone="Africa/Tunis",
                        start_date=cron_start, end_date=cron_end),
            id=f"prewarm_{shift_id}", args=[shift_id], replace_existing=True,
        )
        scheduler.add_job(
            _shift_start,
            CronTrigger(day_of_week=day_str, hour=int(s_hour), minute=int(s_min),
                        timezone="Africa/Tunis",
                        start_date=cron_start, end_date=cron_end),
            id=f"start_{shift_id}", args=[shift_id], replace_existing=True,
        )
        scheduler.add_job(
            _shift_stop,
            CronTrigger(day_of_week=day_str, hour=int(e_hour), minute=int(e_min),
                        timezone="Africa/Tunis",
                        start_date=cron_start, end_date=cron_end),
            id=f"stop_{shift_id}", args=[shift_id], replace_existing=True,
        )
        print(f"[SCHEDULER] Scheduled shift '{shift.get('label', shift_id)}' "
              f"{shift['start_time']}-{shift['end_time']} on {day_str}")


def _reschedule_shift(shift_id):
    """Remove then re-add jobs for a shift (or just remove if inactive/deleted)."""
    _remove_shift_jobs(shift_id)
    if db_writer is None:
        return
    shift = db_writer.get_shift(shift_id)
    if shift and shift.get("active"):
        _schedule_shift(shift)


def _load_all_shift_jobs():
    """Read all active shifts + one-off sessions from DB and schedule them."""
    if db_writer is None:
        print("[SCHEDULER] No DB — skipping shift scheduling")
        return
    # Recurring shifts
    shifts = db_writer.get_all_shifts()
    count = 0
    for s in shifts:
        if s.get("active"):
            _schedule_shift(s)
            count += 1
    # One-off sessions (stored as shifts with type='one_off')
    one_offs = db_writer.get_all_one_off_sessions()
    for oo in one_offs:
        # Reconstruct as a shift-like dict for _schedule_shift
        oo_shift = {
            "id": oo["id"],
            "label": oo.get("label", "One-off"),
            "type": "one_off",
            "start_time": oo["start_time"],
            "end_time": oo["end_time"],
            "session_date": oo.get("date", ""),
            "active": 1,
        }
        _schedule_shift(oo_shift)
        count += 1
    print(f"[SCHEDULER] Loaded {count} active job(s) (shifts + one-offs)")


# ==========================
# MAIN
# ==========================
def _shutdown():
    """Graceful shutdown: close any open stats session so its totals are saved."""
    global _active_session_source, _active_session_group, _active_session_shift_id
    for pid, st in _all_states():
        try:
            if getattr(st, '_stats_active', False):
                print(f"[SHUTDOWN][{pid}] Closing active stats session...")
                st.set_stats_recording(False)
        except Exception as e:
            print(f"[SHUTDOWN][{pid}] Error closing session: {e}")
        try:
            if st.is_running:
                st.stop_processing()
        except Exception:
            pass
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass
    _active_session_source = None
    _active_session_group = None
    _active_session_shift_id = None
    try:
        if db_writer:
            db_writer.stop()
    except Exception:
        pass

atexit.register(_shutdown)
for _sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_sig, lambda s, f: (_shutdown(), exit(0)))


if __name__ == '__main__':
    init_all_pipelines()

    # Start shift scheduler
    _load_all_shift_jobs()
    scheduler.start()
    print("[SCHEDULER] APScheduler started")

    print("\n" + "=" * 60)
    print("  MULTI-PIPELINE WEB SERVER STARTED")
    print(f"  {len(pipelines)} pipeline(s) initialized")
    print("  Video stream + YOLO detection run independently")
    print("=" * 60)
    print(f"  http://localhost:{SERVER_PORT}")
    print(f"  http://196.179.229.162:{SERVER_PORT}/")
    print("=" * 60 + "\n")

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    # Kill any stale process holding our port (prevents "Address already in use")
    try:
        pids = subprocess.check_output(
            ["lsof", "-ti", f":{SERVER_PORT}"], stderr=subprocess.DEVNULL
        ).decode().split()
        my_pid = str(os.getpid())
        for pid in pids:
            pid = pid.strip()
            if pid and pid != my_pid:
                print(f"[STARTUP] Killing stale process {pid} on port {SERVER_PORT}")
                os.kill(int(pid), signal.SIGKILL)
        if pids:
            time.sleep(0.5)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # no process on port, or lsof not available

    # ── gevent WSGI server (no OS-thread-per-request) ──
    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer((SERVER_HOST, SERVER_PORT), app, log=None)
    print(f"[SERVER] gevent WSGIServer listening on {SERVER_HOST}:{SERVER_PORT}")
    http_server.serve_forever()
