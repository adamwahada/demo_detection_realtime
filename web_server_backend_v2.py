"""
Web Server Backend - PARALLEL ARCHITECTURE
import time
import logging
import io

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response, render_template, make_response
from ultralytics import YOLO

from tracking_config import (
    MODEL_PATH, PACKAGE_CLASS_NAME, BARCODE_CLASS_NAME,
    CONFIG, TRACKER_CONFIG, SERVER_HOST, SERVER_PORT,
    CHECKPOINTS, CAMERAS,
    DEFAULT_CHECKPOINT_ID, DEFAULT_CAMERA_ID,
    get_checkpoint, get_camera,
    DEVICE,
)
from tracking_state import TrackingState

# ==========================
# FLASK APP
# ==========================
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ==========================
# GLOBAL STATE
# ==========================
state = TrackingState()

# Track which checkpoint / camera are currently active
current_checkpoint_id = DEFAULT_CHECKPOINT_ID
current_camera_id     = DEFAULT_CAMERA_ID


def init_models(checkpoint_id=None):
    """Load YOLO model on GPU and warm up."""
    global state, current_checkpoint_id
    if checkpoint_id is None:
        checkpoint_id = DEFAULT_CHECKPOINT_ID
    checkpoint = get_checkpoint(checkpoint_id)
    if checkpoint is None:
        raise ValueError(f"Unknown checkpoint id: {checkpoint_id}")

    current_checkpoint_id = checkpoint_id
    print(f"Loading model: {checkpoint['label']} ({checkpoint['path']})...")
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
    print(f"Model loaded on {DEVICE}. mode={state.mode} package={state.package_id} barcode={state.barcode_id} date={state.date_id}")

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
            print(f"Secondary date model loaded: id={state._secondary_date_id}")
        except Exception as e:
            state.secondary_model = None
            state._secondary_date_id = None
            state._use_secondary_date = False
            print(f"Secondary date model load failed (non-fatal): {e}")
    else:
        state.secondary_model = None
        state._secondary_date_id = None
        state._use_secondary_date = False

    # If this checkpoint is the anomaly detector, attempt to register/load AD models
    if state.mode == "anomaly":
        try:
            state._load_ad_models(checkpoint, DEVICE)
        except Exception as e:
            print(f"[AD] _load_ad_models failed: {e}")

    print("Warming up YOLO (dummy inference)...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        state.model(dummy, imgsz=CONFIG["imgsz"], verbose=False)
        import torch
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        print("YOLO warmup complete.")
    except Exception as e:
        print(f"YOLO warmup failed (non-fatal): {e}")


# ==========================
# WEB ROUTES
# ==========================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG stream — serves pre-encoded JPEG bytes. Zero computation here."""
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for stream...", (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    _, buf = cv2.imencode('.jpg', placeholder)
    placeholder_bytes = buf.tobytes()

    def generate():
        while True:
            with state._jpeg_lock:
                jpeg = state._jpeg_bytes
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
    data = request.get_json()
    source = data.get('source', '/dev/video0')
    return jsonify(state.start_processing(source))


@app.route('/api/stop', methods=['POST'])
def api_stop():
    return jsonify(state.stop_processing())


@app.route('/api/pause', methods=['POST'])
def api_pause():
    return jsonify(state.pause_processing())


@app.route('/api/resume', methods=['POST'])
def api_resume():
    return jsonify(state.resume_processing())


@app.route('/api/stats')
def api_stats():
    with state._stats_lock:
        s = dict(state.stats)
    with state._perf_lock:
        perf = dict(state._perf)
    # Enrich with active checkpoint + camera
    s["checkpoint_id"]    = current_checkpoint_id
    s["checkpoint_label"] = (state.current_checkpoint or {}).get("label", "")
    s["checkpoint_mode"]  = state.mode
    s["camera_id"]        = current_camera_id
    s["exit_line_enabled"] = state._exit_line_enabled
    s["exit_line_vertical"] = state._exit_line_vertical
    s["exit_line_inverted"] = state._exit_line_inverted
    s["exit_line_pct"] = state._exit_line_pct
    s["rotation_deg"] = (state._rotation_steps % 4) * 90
    s["perf"] = perf
    s["is_paused"] = getattr(state, '_is_paused', False)
    s["stats_active"] = getattr(state, '_stats_active', False)
    s["session_id"] = getattr(state, '_db_session_id', None)
    s["db_available"] = state._db_writer is not None
    s["db_backend"] = state._db_writer.backend if state._db_writer is not None else None
    s["nok_no_barcode"] = getattr(state, '_nok_no_barcode', 0)
    s["nok_no_date"] = getattr(state, '_nok_no_date', 0)
    return jsonify(s)


@app.route('/api/perf')
def api_perf():
    with state._stats_lock:
        stats = dict(state.stats)
    with state._perf_lock:
        perf = dict(state._perf)

    return jsonify({
        "checkpoint_id": current_checkpoint_id,
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
    total = state.total_packets
    ok_count = state.output_fifo.count("OK")
    nok_count = state.output_fifo.count("NOK")
    return jsonify({
        "stats_active": getattr(state, '_stats_active', False),
        "session_id": getattr(state, '_db_session_id', None),
        "db_available": state._db_writer is not None,
        "db_backend": state._db_writer.backend if state._db_writer is not None else None,
        "total": total,
        "ok_count": ok_count,
        "nok_count": nok_count,
        "nok_no_barcode": getattr(state, '_nok_no_barcode', 0),
        "nok_no_date": getattr(state, '_nok_no_date', 0),
        "nok_rate_pct": round(nok_count / total * 100, 2) if total > 0 else 0.0,
    })


@app.route('/api/stats/toggle', methods=['POST'])
def api_stats_toggle():
    request.get_json(force=True, silent=True)
    result = state.set_stats_recording(not getattr(state, '_stats_active', False))
    result["db_available"] = state._db_writer is not None
    result["db_backend"] = state._db_writer.backend if state._db_writer is not None else None
    return jsonify(result)


@app.route('/api/stats/sessions')
def api_stats_sessions():
    limit = request.args.get('limit', default=50, type=int)
    if state._db_writer is None:
        return jsonify({"sessions": []})
    sessions = state._db_writer.list_sessions(limit=max(1, min(limit, 500)))
    return jsonify({"sessions": sessions})


@app.route('/api/stats/session/<session_id>')
def api_stats_session(session_id):
    if state._db_writer is None:
        return jsonify({}), 404
    data = state._db_writer.get_session_kpis(session_id)
    if not data:
        return jsonify({}), 404
    return jsonify(data)


@app.route('/api/stats/session/<session_id>/snapshots')
def api_stats_session_snapshots(session_id):
    if state._db_writer is None:
        return jsonify({"snapshots": []})
    limit = request.args.get('limit', default=500, type=int)
    rows = state._db_writer.list_snapshots(session_id, limit=max(1, min(limit, 5000)))
    return jsonify({"snapshots": rows})


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        import tracking_state as _ts_mod
        data = request.get_json() or {}
        tracker_changed = False
        for key, value in data.items():
            if key in CONFIG:
                CONFIG[key] = value
            elif key in TRACKER_CONFIG:
                TRACKER_CONFIG[key] = value
                tracker_changed = True
        # If any tracker param changed, rewrite the YAML and update the
        # module-level path so the next model.track() call picks it up.
        if tracker_changed:
            import tempfile, os as _os
            content = "\n".join(f"{k}: {v}" for k, v in TRACKER_CONFIG.items())
            fd, path = tempfile.mkstemp(suffix=".yaml", prefix="bytetrack_")
            with _os.fdopen(fd, "w") as f:
                f.write(content + "\n")
            _ts_mod.TRACKER_YAML_PATH = path
            print(f"[CONFIG] Tracker YAML regenerated → {path}")
        return jsonify({"status": "updated", "config": CONFIG, "tracker_config": TRACKER_CONFIG})
    return jsonify({"config": CONFIG, "tracker_config": TRACKER_CONFIG})


@app.route('/api/fifo')
def api_fifo():
    return jsonify({
        "fifo": state.output_fifo,
        "total_packets": state.total_packets
    })


@app.route('/api/exit_line', methods=['POST'])
def api_exit_line():
    """Toggle exit line on/off. Returns new state."""
    state._exit_line_enabled = not state._exit_line_enabled
    return jsonify({"enabled": state._exit_line_enabled})


@app.route('/api/exit_line_orientation', methods=['POST'])
def api_exit_line_orientation():
    """Toggle exit line between horizontal (False) and vertical (True)."""
    state._exit_line_vertical = not state._exit_line_vertical
    state._recompute_exit_line_y()
    orientation = "vertical" if state._exit_line_vertical else "horizontal"
    print(f'[EXIT LINE] Orientation set to {orientation}')
    return jsonify({"vertical": state._exit_line_vertical, "orientation": orientation})


@app.route('/api/exit_line_invert', methods=['POST'])
def api_exit_line_invert():
    """Toggle direction: % from near edge (False) or far edge (True).
    Use when conveyor runs right-to-left so the slider value stays intuitive.
    """
    state._exit_line_inverted = not state._exit_line_inverted
    state._recompute_exit_line_y()
    print(f'[EXIT LINE] Direction inverted={state._exit_line_inverted}')
    return jsonify({"inverted": state._exit_line_inverted})


@app.route('/api/exit_line_position', methods=['POST'])
def api_exit_line_position():
    """Set exit line position as % from leading edge (5–95). Updates live without restart."""
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
    """Cycle input rotation by 90° counter-clockwise (0/90/180/270)."""
    deg = state.cycle_rotation_ccw()
    return jsonify({"rotation_deg": deg})


@app.route('/api/checkpoints')
def api_checkpoints():
    """List available checkpoints and which one is active."""
    return jsonify({
        "checkpoints": [
            {**cp, "active": cp["id"] == current_checkpoint_id}
            for cp in CHECKPOINTS
        ],
        "active_id": current_checkpoint_id,
    })


@app.route('/api/cameras')
def api_cameras():
    """List available cameras and which one is active."""
    return jsonify({
        "cameras": [
            {**cam, "active": cam["id"] == current_camera_id}
            for cam in CAMERAS
        ],
        "active_id": current_camera_id,
    })


@app.route('/api/switch', methods=['POST'])
def api_switch():
    """
    Switch checkpoint and/or camera.
    """
    global current_checkpoint_id, current_camera_id

    data = request.get_json() or {}
    new_cp_id  = data.get("checkpoint_id")
    new_cam_id = data.get("camera_id")
    custom_src = data.get("custom_source")  # free-text source override

    # Resolve new source
    new_source = None
    if custom_src:
        new_source = custom_src
    elif new_cam_id:
        cam = get_camera(new_cam_id)
        if cam is None:
            return jsonify({"error": f"Unknown camera id: {new_cam_id}"}), 400
        new_source = cam["source"]
        current_camera_id = new_cam_id

    # If only camera changed (not checkpoint), just restart with new source
    if new_cp_id is None or new_cp_id == current_checkpoint_id:
        if new_source and state.is_running:
            state.stop_processing()
            state.start_processing(new_source)
        elif new_source and not state.is_running:
            pass  # source remembered but not started
        return jsonify({
            "status": "camera_switched",
            "checkpoint_id": current_checkpoint_id,
            "camera_id": current_camera_id,
            "source": new_source,
        })

    # Checkpoint switch: unload old, load new, restart if was running
    checkpoint = get_checkpoint(new_cp_id)
    if checkpoint is None:
        return jsonify({"error": f"Unknown checkpoint id: {new_cp_id}"}), 400

    # Override source if a new camera was also selected
    was_running = state.is_running
    prev_source = state.video_source

    result = state.switch_checkpoint(checkpoint)
    current_checkpoint_id = new_cp_id

    # Use new_source if provided, otherwise keep existing source
    target_source = new_source or prev_source
    if was_running and target_source:
        state.start_processing(target_source)

    result["source"] = target_source
    result["camera_id"] = current_camera_id
    return jsonify(result)

# ==========================
# STATS / DB ENDPOINTS
# ==========================

@app.route('/api/stats/status')
def api_stats_status():
    """Return current stats recording status + live KPI counters."""
    ok  = state.output_fifo.count("OK")
    nok = state.output_fifo.count("NOK")
    total = state.total_packets
    return jsonify({
        "stats_active":      getattr(state, '_stats_active', True),
        "session_id":        getattr(state, '_db_session_id', None),
        "db_available":      state._db_writer is not None,
        "total":             total,
        "ok_count":          ok,
        "nok_count":         nok,
        "nok_no_barcode":    getattr(state, '_nok_no_barcode', 0),
        "nok_no_date":       getattr(state, '_nok_no_date', 0),
        "nok_both":          getattr(state, '_nok_both', 0),
        "nok_rate_pct":      round(nok / total * 100, 2) if total > 0 else 0.0,
    })


@app.route('/api/stats/toggle', methods=['POST'])
def api_stats_toggle():
    """Toggle stats recording ON/OFF.
    ON  → open new DB session, reset all counters, start recording.
    OFF → close session with final totals, reset counters to 0, session_id = None.
    """
    request.get_json(force=True, silent=True)
    currently_active = getattr(state, '_stats_active', False)
    new_state = not currently_active

    if new_state:
        # ── Turning ON: reset counters then open a fresh session ──
        state.total_packets   = 0
        state.output_fifo     = []
        state.packet_numbers  = {}
        state._nok_no_barcode = 0
        state._nok_no_date    = 0
        state._nok_both       = 0
        with state._stats_lock:
            state.stats["total_packets"] = 0
            state.stats["packages_ok"]   = 0
            state.stats["packages_nok"]  = 0
            state.stats["fifo_queue"]    = []
        new_sid = None
        if state._db_writer:
            cp_id   = (state.current_checkpoint or {}).get("id", "")
            cam_src = str(state.video_source or "")
            new_sid = state._db_writer.open_session(
                checkpoint_id=cp_id,
                camera_source=cam_src,
            )
            state._db_session_id = new_sid
            state._db_writer.set_active(True)
        state._stats_active = True
        return jsonify({"stats_active": True, "session_id": new_sid})

    else:
        # ── Turning OFF: close session, reset all counters ──
        if state._db_writer and state._db_session_id:
            state._db_writer.close_session(
                state._db_session_id,
                totals={
                    "total":          state.total_packets,
                    "ok_count":       state.output_fifo.count("OK"),
                    "nok_no_barcode": getattr(state, '_nok_no_barcode', 0),
                    "nok_no_date":    getattr(state, '_nok_no_date', 0),
                    "nok_both":       getattr(state, '_nok_both', 0),
                }
            )
            if state._db_writer:
                state._db_writer.set_active(False)
        # Reset everything to 0
        state._db_session_id  = None
        state._stats_active   = False
        state.total_packets   = 0
        state.output_fifo     = []
        state.packet_numbers  = {}
        state._nok_no_barcode = 0
        state._nok_no_date    = 0
        state._nok_both       = 0
        with state._stats_lock:
            state.stats["total_packets"] = 0
            state.stats["packages_ok"]   = 0
            state.stats["packages_nok"]  = 0
            state.stats["fifo_queue"]    = []
        return jsonify({"stats_active": False, "session_id": None})


@app.route('/api/stats/reset', methods=['POST'])
def api_stats_reset():
    """
    Close current session (writes final totals to DB) and open a new one.
    Zeroes all in-memory counters. Detection keeps running.
    """
    request.get_json(force=True, silent=True)  # consume body if any
    # 1. Close current session with final totals
    if state._db_writer and state._db_session_id:
        state._db_writer.close_session(
            state._db_session_id,
            totals={
                "total":          state.total_packets,
                "ok_count":       state.output_fifo.count("OK"),
                "nok_no_barcode": getattr(state, '_nok_no_barcode', 0),
                "nok_no_date":    getattr(state, '_nok_no_date', 0),
                "nok_both":       getattr(state, '_nok_both', 0),
            }
        )

    # 2. Zero in-memory counters (without stopping threads)
    state.total_packets  = 0
    state.output_fifo    = []
    state.packet_numbers = {}
    state._nok_no_barcode = 0
    state._nok_no_date    = 0
    state._nok_both       = 0
    # Note: packages and packets_crossed_line intentionally NOT reset
    # so live tracking isn't disrupted mid-frame.

    # 3. Open new session
    new_session_id = None
    if state._db_writer:
        cp_id  = (state.current_checkpoint or {}).get("id", "")
        cam_src = state.video_source or ""
        new_session_id = state._db_writer.open_session(
            checkpoint_id=cp_id,
            camera_source=str(cam_src),
        )
        state._db_session_id = new_session_id

    with state._stats_lock:
        state.stats["total_packets"] = 0
        state.stats["packages_ok"]   = 0
        state.stats["packages_nok"]  = 0
        state.stats["fifo_queue"]    = []

    return jsonify({"status": "reset", "new_session_id": new_session_id})


@app.route('/api/sessions')
def api_sessions():
    """List recent sessions from the database."""
    if not state._db_writer:
        return jsonify({"error": "DB not available"}), 503
    limit = int(request.args.get('limit', 50))
    sessions = state._db_writer.list_sessions(limit=limit)
    # Convert datetime objects to ISO strings for JSON serialisation
    for s in sessions:
        for k in ('started_at', 'ended_at'):
            if s.get(k) and hasattr(s[k], 'isoformat'):
                s[k] = s[k].isoformat()
    return jsonify({"sessions": sessions, "count": len(sessions)})


@app.route('/api/export/csv')
def api_export_csv():
    """Download a session summary as CSV. ?session_id=<uuid>"""
    if not state._db_writer:
        return jsonify({"error": "DB not available"}), 503
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    csv_data = state._db_writer.export_csv(session_id)
    response = make_response(csv_data)
    response.headers['Content-Type']        = 'text/csv; charset=utf-8'
    response.headers['Content-Disposition'] = f'attachment; filename="session_{session_id[:8]}.csv"'
    return response


@app.route('/api/export/json')
def api_export_json():
    """Download a session summary as JSON. ?session_id=<uuid>"""
    if not state._db_writer:
        return jsonify({"error": "DB not available"}), 503
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    kpis = state._db_writer.get_session_kpis(session_id)
    if not kpis:
        return jsonify({"error": "Session not found"}), 404
    total = kpis.get('total', 0)
    ok    = kpis.get('ok_count', 0)
    kpis['nok_count'] = total - ok
    kpis['nok_rate']  = round((total - ok) / total * 100, 2) if total > 0 else 0.0
    return jsonify(kpis)


# ==========================
# MAIN
# ==========================
if __name__ == '__main__':
    init_models()
    print("\n" + "=" * 60)
    print("  PARALLEL WEB SERVER STARTED")
    print("  Video stream + YOLO detection run independently")
    print("=" * 60)
    print(f"  http://localhost:{SERVER_PORT}")
    print(f"  http://196.179.229.162:{SERVER_PORT}/")
    print("=" * 60 + "\n")

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)
