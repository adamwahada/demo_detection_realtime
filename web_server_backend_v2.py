import time
import logging
import io

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response, render_template, make_response, send_from_directory
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
    # move model to configured device (best-effort)
    try:
        state.model.to(DEVICE)
    except Exception:
        pass
    # attempt FP16 conversion when using CUDA
    try:
        import torch
        if str(DEVICE).lower().startswith("cuda") and torch.cuda.is_available():
            try:
                getattr(state.model, "model", None).half()
                print("[INIT] Converted model to FP16 (half precision).")
            except Exception as e:
                print(f"[INIT] FP16 conversion failed: {e}")
    except Exception:
        pass
    print(f"Model device: {state.model.device}")
    if torch.cuda.is_available():
        print(f"CUDA memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
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

    # Load EfficientAD models if anomaly mode
    if state.mode == "anomaly":
        state._load_ad_models(checkpoint, DEVICE)

    # Apply default rotation for this checkpoint
    default_rot = checkpoint.get("default_rotation", 0) % 4
    state._rotation_steps = default_rot
    deg = default_rot * 90
    with state._stats_lock:
        state.stats["rotation_deg"] = deg
    print(f"Default rotation: {deg}° CCW")

    print("Warming up YOLO (dummy inference)...")
    dummy = np.zeros((640, 640, 3), dtype=np.float32)
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
            time.sleep(0.033)

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


ANOMALY_DEMO_VIDEOS = {'testanomalie.mp4', 'testanomalie2.mp4'}

@app.route('/api/start', methods=['POST'])
def api_start():
    import os as _os
    data = request.get_json()
    source = data.get('source', '/dev/video0')

    # Auto-configure for pre-rendered anomaly demo videos
    is_anomaly_demo = (
        isinstance(source, str) and
        _os.path.basename(source).lower() in ANOMALY_DEMO_VIDEOS
    )
    if is_anomaly_demo:
        state._raw_mode = True
        state._exit_line_enabled = False

    result = state.start_processing(source)
    result['anomaly_demo_mode'] = is_anomaly_demo
    return jsonify(result)


@app.route('/api/stop', methods=['POST'])
def api_stop():
    return jsonify(state.stop_processing())


@app.route('/api/pause', methods=['POST'])
def api_pause():
    """Pause video playback (threads stay alive, cap stays open)."""
    return jsonify(state.pause_processing())


@app.route('/api/resume', methods=['POST'])
def api_resume():
    """Resume from pause."""
    return jsonify(state.resume_processing())


@app.route('/api/seek', methods=['POST'])
def api_seek():
    """Seek video to a position. Body: {"position": 50} (0-100%)."""
    data = request.get_json() or {}
    pct = data.get('position', 0)
    return jsonify(state.seek_video(pct))


@app.route('/api/speed', methods=['POST'])
def api_speed():
    """Set playback speed. Body: {"speed": 1.0} (0.1–4.0)."""
    data = request.get_json() or {}
    speed = data.get('speed', 1.0)
    return jsonify(state.set_playback_speed(speed))


@app.route('/api/raw_mode', methods=['POST'])
def api_raw_mode():
    """Toggle raw mode (video without detection overlay)."""
    state._raw_mode = not state._raw_mode
    mode = "raw" if state._raw_mode else "detection"
    print(f'[RAW MODE] {mode}')
    return jsonify({"raw_mode": state._raw_mode})


@app.route('/api/stats')
def api_stats():
    with state._stats_lock:
        s = dict(state.stats)
    # Enrich with active checkpoint + camera
    s["checkpoint_id"]    = current_checkpoint_id
    s["checkpoint_label"] = (state.current_checkpoint or {}).get("label", "")
    s["checkpoint_mode"]  = state.mode
    s["camera_id"]        = current_camera_id
    s["exit_line_enabled"] = state._exit_line_enabled
    s["exit_line_vertical"] = state._exit_line_vertical
    s["exit_line_inverted"] = state._exit_line_inverted
    s["rotation_deg"] = (state._rotation_steps % 4) * 90
    # Current exit line as % from leading edge (for slider sync)
    s["exit_line_pct"] = state._exit_line_pct
    # Video playback info
    s["paused"] = state._paused
    s["has_source"] = bool(getattr(state, 'video_source', None))
    s["playback_speed"] = state._playback_speed
    s["raw_mode"] = state._raw_mode
    s["is_video_file"] = state._is_video_file
    s["video_pos_frames"] = state._video_pos_frames
    s["video_total_frames"] = state._video_total_frames
    vfps = state._video_fps
    s["video_pos_sec"] = round(state._video_pos_frames / vfps, 1) if vfps > 0 else 0
    s["video_duration_sec"] = round(state._video_total_frames / vfps, 1) if vfps > 0 else 0
    return jsonify(s)


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
    state._recompute_exit_line_y()  # recompute with new ref dimension
    orientation = "vertical" if state._exit_line_vertical else "horizontal"
    print(f'[EXIT LINE] Orientation set to {orientation}')
    return jsonify({"vertical": state._exit_line_vertical, "orientation": orientation})


@app.route('/api/exit_line_invert', methods=['POST'])
def api_exit_line_invert():
    """Toggle direction: % measured from near edge (False) or far edge (True).
    Use this when the conveyor moves in the opposite direction so that the
    slider value makes intuitive sense (e.g. 85% from the right instead of left).
    """
    state._exit_line_inverted = not state._exit_line_inverted
    state._recompute_exit_line_y()
    inv = state._exit_line_inverted
    print(f'[EXIT LINE] Direction inverted={inv}')
    return jsonify({"inverted": inv})


@app.route('/api/exit_line_position', methods=['POST'])
def api_exit_line_position():
    """Set exit line position as % from top (0–100). Updates live without restart."""
    data = request.get_json() or {}
    pct = data.get('position', 85)
    try:
        pct = max(5, min(95, int(pct)))
    except (TypeError, ValueError):
        return jsonify({'error': 'position must be an integer 0-100'}), 400

    # Update config ratio + live state percentage
    CONFIG['exit_line_ratio'] = round(1.0 - pct / 100.0, 4)
    state._exit_line_pct = pct

    # Apply immediately if video is running (frame height known), else queued for next Start
    if state._frame_height > 0:
        state._recompute_exit_line_y()
        print(f'[EXIT LINE] Position set to {pct}% (y/x={state._exit_line_y}, rot={state._rotation_steps*90}\u00b0 CCW)')
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

    # Checkpoint switch: unload old, load new. Restart only if it was
    # running and NOT paused (respect user-initiated pause).
    checkpoint = get_checkpoint(new_cp_id)
    if checkpoint is None:
        return jsonify({"error": f"Unknown checkpoint id: {new_cp_id}"}), 400

    # Override source if a new camera was also selected
    was_running = state.is_running
    was_paused = getattr(state, '_paused', False)
    prev_source = state.video_source

    result = state.switch_checkpoint(checkpoint)
    current_checkpoint_id = new_cp_id

    # Use new_source if provided, otherwise keep existing source
    target_source = new_source or prev_source
    # Only auto-restart if it was running and the user hadn't paused it.
    if was_running and target_source and not was_paused:
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


@app.route('/api/sessions/<session_id>/images')
def api_session_images(session_id):
    """Get all images for a session."""
    if not state._db_writer:
        return jsonify({"error": "DB not available"}), 503
    images = state._db_writer.get_session_images(session_id)
    # Convert datetime objects to ISO strings
    for img in images:
        if img.get('captured_at') and hasattr(img['captured_at'], 'isoformat'):
            img['captured_at'] = img['captured_at'].isoformat()
    return jsonify({"images": images, "count": len(images)})


@app.route('/api/daily-stats')
def api_daily_stats():
    """Get daily aggregated statistics for today or a specific date."""
    if not state._db_writer:
        return jsonify({"error": "DB not available"}), 503
    date = request.args.get('date')
    stats = state._db_writer.get_daily_stats(date)
    return jsonify(stats)


@app.route('/images/<path:filepath>')
def serve_image(filepath):
    """Serve saved images from the images directory."""
    try:
        return send_from_directory(_os.path.join(_os.path.dirname(__file__), 'images'), filepath)
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404


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
