"""
Configuration for the Parallel Tracking Web Server.
All tunable parameters are centralized here.
"""

# ==========================
# MODEL
# ==========================
MODEL_PATH = "/home/adam/YOLO_Benchmark/ultralytics/base_models/yolo26-BB(farine+paquet).pt"
PACKAGE_CLASS_NAME = "package"
BARCODE_CLASS_NAME = "barcode"

# ==========================
# DETECTION & TRACKING
# ==========================
CONFIG = {
    "conf_paquet": 0.5,
    "conf_barcode": 0.5,
    "imgsz": 416,
    "track_thresh": 0.5,
    "track_buffer": 60,
    "match_thresh": 0.8,
    "exit_line_ratio": 0.15,
    "exit_line_proximity": 50,
}

# ==========================
# CAMERA DEFAULTS
# ==========================
CAMERA_FPS = 30
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ==========================
# VIDEO FILE EXTENSIONS
# ==========================
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')

# ==========================
# COMPOSITOR
# ==========================
JPEG_QUALITY = 80

# ==========================
# SERVER
# ==========================
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5000

# ==========================
# FRAME SKIP (detector receives 1 frame out of N)
# ==========================
DETECTOR_FRAME_SKIP = 3
