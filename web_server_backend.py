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

import atexit
import json
import re
import signal
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response, render_template, send_file
from flask_cors import CORS
from ultralytics import YOLO

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.base import JobLookupError

from tracking_config import (
    MODEL_PATH, PACKAGE_CLASS_NAME, BARCODE_CLASS_NAME,
    CONFIG, SERVER_HOST, SERVER_PORT,
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
CORS(app, resources={r"/api/*": {"origins": "*"},
                     r"/video_feed": {"origins": "*"}})

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


@app.route('/api/stats/session/<session_id>/crossings')
def api_stats_session_crossings(session_id):
    if state._db_writer is None:
        return jsonify({"crossings": []})
    limit = request.args.get('limit', default=5000, type=int)
    rows = state._db_writer.list_crossings(session_id, limit=max(1, min(limit, 10000)))
    return jsonify({"crossings": rows})


@app.route('/api/proof/<session_id>/<defect_type>/<int:packet_num>')
def api_proof_image(session_id, defect_type, packet_num):
    """Serve a proof image for a defective packet."""
    if defect_type not in ("nobarcode", "nodate", "anomaly"):
        return jsonify({"error": "invalid defect_type"}), 400
    base = Path(__file__).parent / "liveImages" / session_id / defect_type
    img_path = base / f"packet_{packet_num}.png"
    if not img_path.is_file():
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
# SHIFTS CRUD
# ==========================
_HH_MM_RE = re.compile(r'^([01]\d|2[0-3]):[0-5]\d$')
_VALID_DAYS = {'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'}


def _validate_time(value):
    return isinstance(value, str) and _HH_MM_RE.match(value)


def _validate_days(value):
    if not isinstance(value, list) or len(value) == 0:
        return False
    return all(isinstance(d, str) and d.lower() in _VALID_DAYS for d in value)


@app.route('/api/shifts', methods=['GET'])
def api_shifts_list():
    if state._db_writer is None:
        return jsonify({"shifts": []})
    shifts = state._db_writer.get_all_shifts()
    return jsonify({"shifts": shifts})


@app.route('/api/shifts', methods=['POST'])
def api_shifts_create():
    if state._db_writer is None:
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

    days_of_week = data.get("days_of_week", [])
    if not _validate_days(days_of_week):
        return jsonify({"error": "days_of_week must be a non-empty array of mon-sun"}), 400

    camera_source = str(data.get("camera_source", "0"))
    checkpoint_id = data.get("checkpoint_id", "tracking")
    if get_checkpoint(checkpoint_id) is None:
        return jsonify({"error": f"unknown checkpoint_id: {checkpoint_id}"}), 400

    shift = {
        "id": str(uuid.uuid4()),
        "label": label,
        "start_time": start_time,
        "end_time": end_time,
        "days_of_week": json.dumps([d.lower() for d in days_of_week]),
        "camera_source": camera_source,
        "checkpoint_id": checkpoint_id,
        "active": 1,
        "created_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
    }
    ok = state._db_writer.insert_shift(shift)
    if not ok:
        return jsonify({"error": "insert failed"}), 500

    _reschedule_shift(shift["id"])
    return jsonify({"shift": shift}), 201


@app.route('/api/shifts/<shift_id>', methods=['PUT'])
def api_shifts_update(shift_id):
    if state._db_writer is None:
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
    if "days_of_week" in data:
        if not _validate_days(data["days_of_week"]):
            return jsonify({"error": "days_of_week must be a non-empty array of mon-sun"}), 400
        fields["days_of_week"] = json.dumps([d.lower() for d in data["days_of_week"]])
    if "camera_source" in data:
        fields["camera_source"] = str(data["camera_source"])
    if "checkpoint_id" in data:
        if get_checkpoint(data["checkpoint_id"]) is None:
            return jsonify({"error": f"unknown checkpoint_id: {data['checkpoint_id']}"}), 400
        fields["checkpoint_id"] = data["checkpoint_id"]

    if not fields:
        return jsonify({"error": "no valid fields to update"}), 400

    ok = state._db_writer.update_shift(shift_id, fields)
    if not ok:
        return jsonify({"error": "update failed"}), 500

    _reschedule_shift(shift_id)
    updated = state._db_writer.get_shift(shift_id)
    return jsonify({"shift": updated})


@app.route('/api/shifts/<shift_id>', methods=['DELETE'])
def api_shifts_delete(shift_id):
    if state._db_writer is None:
        return jsonify({"error": "database not available"}), 503
    _remove_shift_jobs(shift_id)
    ok = state._db_writer.delete_shift(shift_id)
    if not ok:
        return jsonify({"error": "delete failed"}), 500
    return jsonify({"deleted": True})


@app.route('/api/shifts/<shift_id>/toggle', methods=['POST'])
def api_shifts_toggle(shift_id):
    if state._db_writer is None:
        return jsonify({"error": "database not available"}), 503
    new_active = state._db_writer.toggle_shift(shift_id)
    if new_active is None:
        return jsonify({"error": "toggle failed"}), 500
    _reschedule_shift(shift_id)
    return jsonify({"id": shift_id, "active": new_active})


# ==========================
# SCHEDULER
# ==========================
scheduler = BackgroundScheduler(daemon=True)


def _shift_start(shift_id):
    """Called by APScheduler when a shift's start_time fires."""
    global current_checkpoint_id
    if state._db_writer is None:
        return
    shift = state._db_writer.get_shift(shift_id)
    if not shift or not shift.get("active"):
        return

    label = shift.get("label", shift_id)
    cp_id = shift.get("checkpoint_id", "tracking")
    cam_src = shift.get("camera_source", "0")
    print(f"[SCHEDULER] Shift '{label}' starting (checkpoint={cp_id}, camera={cam_src})")

    # Switch checkpoint if needed
    if cp_id != current_checkpoint_id:
        cp = get_checkpoint(cp_id)
        if cp is not None:
            result = state.switch_checkpoint(cp)
            current_checkpoint_id = cp_id
            print(f"[SCHEDULER] Checkpoint switched: {result.get('status')}")

    # Start processing
    if not state.is_running:
        state.start_processing(cam_src)
    # Start stats recording (opens DB session)
    if not getattr(state, '_stats_active', False):
        state.set_stats_recording(True)
    print(f"[SCHEDULER] Shift '{label}' started automatically")


def _shift_stop(shift_id):
    """Called by APScheduler when a shift's end_time fires."""
    if state._db_writer is None:
        return
    shift = state._db_writer.get_shift(shift_id)
    label = shift.get("label", shift_id) if shift else shift_id
    print(f"[SCHEDULER] Shift '{label}' stopping")

    # Stop stats recording (closes DB session with totals)
    if getattr(state, '_stats_active', False):
        state.set_stats_recording(False)
    # Stop processing (camera + threads)
    if state.is_running:
        state.stop_processing()
    print(f"[SCHEDULER] Shift '{label}' stopped automatically")


def _remove_shift_jobs(shift_id):
    """Remove start/stop jobs for a shift (if they exist)."""
    for prefix in ("start_", "stop_"):
        try:
            scheduler.remove_job(f"{prefix}{shift_id}")
        except JobLookupError:
            pass


def _schedule_shift(shift):
    """Add start + stop cron jobs for one active shift."""
    shift_id = shift["id"]
    days_raw = shift["days_of_week"]
    if isinstance(days_raw, str):
        days_raw = json.loads(days_raw)
    day_str = ",".join(d.lower() for d in days_raw)

    s_hour, s_min = shift["start_time"].split(":")
    e_hour, e_min = shift["end_time"].split(":")

    scheduler.add_job(
        _shift_start,
        CronTrigger(day_of_week=day_str, hour=int(s_hour), minute=int(s_min)),
        id=f"start_{shift_id}",
        args=[shift_id],
        replace_existing=True,
    )
    scheduler.add_job(
        _shift_stop,
        CronTrigger(day_of_week=day_str, hour=int(e_hour), minute=int(e_min)),
        id=f"stop_{shift_id}",
        args=[shift_id],
        replace_existing=True,
    )
    print(f"[SCHEDULER] Scheduled shift '{shift.get('label', shift_id)}' "
          f"{shift['start_time']}-{shift['end_time']} on {day_str}")


def _reschedule_shift(shift_id):
    """Remove then re-add jobs for a shift (or just remove if inactive/deleted)."""
    _remove_shift_jobs(shift_id)
    if state._db_writer is None:
        return
    shift = state._db_writer.get_shift(shift_id)
    if shift and shift.get("active"):
        _schedule_shift(shift)


def _load_all_shift_jobs():
    """Read all active shifts from DB and schedule them. Called once at startup."""
    if state._db_writer is None:
        print("[SCHEDULER] No DB — skipping shift scheduling")
        return
    shifts = state._db_writer.get_all_shifts()
    count = 0
    for s in shifts:
        if s.get("active"):
            _schedule_shift(s)
            count += 1
    print(f"[SCHEDULER] Loaded {count} active shift(s)")


# ==========================
# MAIN
# ==========================
def _shutdown():
    """Graceful shutdown: close any open stats session so its totals are saved."""
    try:
        if getattr(state, '_stats_active', False):
            print("[SHUTDOWN] Closing active stats session...")
            state.set_stats_recording(False)
    except Exception as e:
        print(f"[SHUTDOWN] Error closing session: {e}")
    try:
        if state.is_running:
            state.stop_processing()
    except Exception:
        pass
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass
    try:
        if state._db_writer:
            state._db_writer.stop()
    except Exception:
        pass

atexit.register(_shutdown)
for _sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_sig, lambda s, f: (_shutdown(), exit(0)))


if __name__ == '__main__':
    init_models()

    # Start shift scheduler
    _load_all_shift_jobs()
    scheduler.start()
    print("[SCHEDULER] APScheduler started")

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
