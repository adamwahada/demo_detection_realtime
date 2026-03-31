"""
Configuration for the Parallel Tracking Web Server.
All tunable parameters are centralized here.
"""

# ==========================
# CHECKPOINTS  (add or remove models here)
# ==========================
# mode = "tracking" → full ByteTrack + barcode logic
# mode = "date"     → simple all-class detection overlay (no tracking)
CHECKPOINTS = [
    {
        "id":            "tracking",
        "label":         "Tracking Paquet+Detection",
        "path":          "bestexp2.pt",
        "mode":          "tracking",
        "package_class": "package",
        "barcode_class": "barcode",
        "date_class":    None,
    },
    {
        "id":            "date",
        "label":         "Date Detection",
        "path":          "yolo26-BB(date).pt",
        "mode":          "date",
        "package_class": None,   
        "barcode_class": None,
        "date_class":    "date",
    },
    {
        "id":            "barcode_date",
        "label":         "Tracking Paquet+Barcode+Date",
        "path":          "yolo26m_BB_barcode_date.pt",
        "mode":          "tracking",
        "package_class": "package",
        "barcode_class": "barcode",
        "date_class":    "date",
        "require_date_for_ok": True,
        # Exit line defaults for this checkpoint (vertical line at 85%, near the left, inverted)
        "exit_line_pct": 85,
        "exit_line_vertical": True,
        "exit_line_inverted": True,
        # Secondary model for maximum date-detection accuracy.
        # Runs in parallel on each frame; its date detections are used
        # for OK/NOK validation alongside barcodes from the primary model.
        "secondary_date_model_path": "yolo26-BB(date).pt",
        "secondary_date_class":      "date",
    },
    {
        "id":            "anomaly",
        "label":         "Segmentation + Anomaly Detection",
        "path":          "yolo26m_seg_farine_FV.pt",
        "mode":          "anomaly",
        "package_class": "farine",
        "barcode_class": None,
        # YOLO overrides for segmentation quality
        "yolo_imgsz":    640,
        "yolo_conf":     0.5,
        # EfficientAD model paths
        "ad_teacher":    "teacher_best.pth",
        "ad_student":    "student_best.pth",
        "ad_autoencoder": "autoencoder_best.pth",
        # Anomaly detection parameters
        "ad_thresh":     5000.0,
        "ad_imgsz":      256,
        "ad_strategy":   "MAJORITY",
        "ad_margin_pct": 0.1,
        "ad_erosion_size": 3,
        "ad_max_scans":  5,
        # Zone: start/end scanning as % of frame width
        # Packets flow RIGHT → LEFT:
        #   zone_end_pct   = ENTRY line (right side, where scanning begins)
        #   zone_start_pct = EXIT  line (left side, decision is locked & queued)
        "zone_start_pct": 0.20,
        "zone_end_pct":   0.60,
    },
]

# Active checkpoint at startup (must match one of the ids above)
DEFAULT_CHECKPOINT_ID = "tracking"

# ==========================
# CAMERAS  (add your camera sources here)
# ==========================
CAMERAS = [
    {"id": "cam0", "label": "Camera 0",  "source": 0},
    {"id": "cam1", "label": "Camera 1",  "source": 1},
]

DEFAULT_CAMERA_ID = "cam0"

# ==========================
# PIPELINES  (each pipeline = one camera + one checkpoint running in parallel)
# ==========================
PIPELINES = [
    {"id": "pipeline_0", "label": "Tracking Line",  "camera_source": "/home/adam/YOLO_Benchmark/ultralytics/V1_comp.mp4", "checkpoint_id": "barcode_date"},
    {"id": "pipeline_1", "label": "Anomaly Line",   "camera_source": "/home/adam/YOLO_Benchmark/ultralytics/testAnomalie-Trim3.mp4", "checkpoint_id": "anomaly"},
]

DEFAULT_VIEW_PIPELINE = "pipeline_0"

# ==========================
# LEGACY alias (kept for imports that still use MODEL_PATH)
# ==========================
MODEL_PATH          = CHECKPOINTS[0]["path"]
PACKAGE_CLASS_NAME  = CHECKPOINTS[0]["package_class"]
BARCODE_CLASS_NAME  = CHECKPOINTS[0]["barcode_class"]

# ==========================
# HELPER LOOKUPS
# ==========================
def get_checkpoint(checkpoint_id):
    for cp in CHECKPOINTS:
        if cp["id"] == checkpoint_id:
            return cp
    return None

def get_camera(camera_id):
    for cam in CAMERAS:
        if cam["id"] == camera_id:
            return cam
    return None

DEVICE = 'cuda'  
# ==========================
# DETECTION & TRACKING
# ==========================
CONFIG = {
    "conf_paquet": 0.45,       
    "conf_barcode": 0.45,
    "conf_date": 0.30,
    "imgsz": 640,  
    "barcode_match_iou_min": 0.01,
    "date_match_iou_min": 0.01,
    "barcode_match_inside_min": 0.60,
    "date_match_inside_min": 0.60,
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
# Set to 1 to disable skipping — RTX 4090 can run inference on every frame
# at full speed. Skipping frames breaks Ultralytics persist=True tracker
# because its Kalman filter expects consecutive input.
# ==========================
DETECTOR_FRAME_SKIP = 1

# Anomaly mode processes heavier per-frame (segmentation + EfficientAD),
# so skip more frames to keep the pipeline responsive.
# 3 means detector runs at ~10fps on a 30fps source.
ANOMALY_FRAME_SKIP = 3

 
# ==========================
# BYTETRACK TRACKER (built-in Ultralytics)
# ==========================
TRACKER_CONFIG = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.45,
    "new_track_thresh": 0.6,
    "track_buffer": 60,
    "match_thresh": 0.8,
    "fuse_score": True,
}