"""
Web Server Backend - PARALLEL ARCHITECTURE
===========================================
Entry point: Flask routes + model init.
All logic is split into separate modules:
  - tracking_config.py  : all tunable parameters
  - helpers.py          : bbox metrics & fallen detection
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
    CONFIG, SERVER_HOST, SERVER_PORT,
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


def init_models():
    """Load YOLO model on GPU and warm up."""
    global state
    print("Loading YOLO model...")
    state.model = YOLO(MODEL_PATH)
    state.model.to('cuda')
    names = state.model.names
    state.package_id = [k for k, v in names.items() if v == PACKAGE_CLASS_NAME][0]
    state.barcode_id = [k for k, v in names.items() if v == BARCODE_CLASS_NAME][0]
    print(f"Model loaded on GPU. Package={state.package_id}, Barcode={state.barcode_id}")

    print("Warming up YOLO (dummy inference)...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        state.model(dummy, imgsz=CONFIG["imgsz"], verbose=False)
        import torch
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


@app.route('/api/stats')
def api_stats():
    with state._stats_lock:
        return jsonify(dict(state.stats))


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
