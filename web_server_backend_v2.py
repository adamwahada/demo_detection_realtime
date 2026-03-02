"""
Web Server Backend - PARALLEL ARCHITECTURE
===========================================
Entry point: Flask routes + model init.
All logic is split into separate modules:
  - tracking_config.py  : all tunable parameters
  - helpers.py          : bbox metrics
  - tracking_state.py   : TrackingState (reader, detector, compositor threads)
  - templates/index.html: web UI
"""

import time
import logging

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response, render_template
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

    # Apply default rotation for this checkpoint
    default_rot = checkpoint.get("default_rotation", 0) % 4
    state._rotation_steps = default_rot
    deg = default_rot * 90
    with state._stats_lock:
        state.stats["rotation_deg"] = deg
    print(f"Default rotation: {deg}° CCW")

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
    s["rotation_deg"] = (state._rotation_steps % 4) * 90
    # Current exit line as % from leading edge (for slider sync)
    s["exit_line_pct"] = state._exit_line_pct
    # Video playback info
    s["paused"] = state._paused
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
    Body: { "checkpoint_id": "date", "camera_id": "cam1" }
    Either key is optional. Switching unloads the current model from VRAM
    and loads the new one. If a camera_id is given and no source is provided
    the camera source from CAMERAS config is used, unless a custom
    'custom_source' key is supplied.
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
