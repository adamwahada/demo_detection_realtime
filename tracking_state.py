"""
TrackingState — Parallel architecture for video reading, YOLO detection, and compositing.
Thread 1 (Reader):   Reads video at native FPS — smooth, never waits for YOLO
Thread 2 (Detector): Runs YOLO + ByteTrack in parallel — updates stats & overlays
Thread 3 (Compositor): Composites raw frame + detection overlay, encodes JPEG
"""

import os
import time
import threading
from pathlib import Path

import cv2
import numpy as np

from helpers import calculate_bbox_metrics, letterbox_image
from tracking_config import (
    CONFIG, VIDEO_EXTENSIONS, JPEG_QUALITY,
    CAMERA_FPS, CAMERA_WIDTH, CAMERA_HEIGHT,
    DETECTOR_FRAME_SKIP,
    TRACKER_CONFIG,
)

try:
    from db_writer import DBWriter
    from db_config import SNAPSHOT_EVERY_N_PACKETS
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    SNAPSHOT_EVERY_N_PACKETS = 50


def _write_tracker_yaml():
    """Write TRACKER_CONFIG dict to a temp YAML file (Ultralytics needs a file path)."""
    import tempfile
    content = "\n".join(f"{k}: {v}" for k, v in TRACKER_CONFIG.items())
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="bytetrack_")
    with os.fdopen(fd, "w") as f:
        f.write(content + "\n")
    return path


TRACKER_YAML_PATH = _write_tracker_yaml()


class TrackingState:
    """
    Three independent threads:
      - Reader:     captures frames at native FPS (smooth video, no YOLO dependency)
      - Detector:   runs YOLO + ByteTrack on latest frame, updates overlay & stats
      - Compositor: composites raw frame + detection overlay, encodes JPEG

    The video feed composites the latest raw frame with the latest detection
    overlay, so the stream is always smooth regardless of YOLO speed.
    """

    def __init__(self):
        self.model = None
        self.package_id = None
        self.barcode_id = None
        self.date_id = None

        # Secondary date-detection model (loaded when checkpoint has
        # "secondary_date_model_path"). Runs in parallel for best accuracy.
        self.secondary_model = None
        self._secondary_date_id = None
        self._use_secondary_date = False  # True when secondary model is active

        # Active checkpoint info (set by switch_checkpoint / init_models)
        self.mode = "tracking"          # "tracking", "date", or "anomaly"
        self.current_checkpoint = None  # the checkpoint dict from CHECKPOINTS

        # ── EfficientAD anomaly detection models (loaded for mode="anomaly") ──
        self._ad_teacher = None
        self._ad_student = None
        self._ad_autoencoder = None
        self._ad_mean = None
        self._ad_std = None
        self._ad_quantiles = None
        self._ad_transform = None
        self._ad_track_states = {}  # {track_id: {'results': [], 'decision': None}}

        # Session generation counter — prevents old reader threads from
        # clobbering is_running after a camera/checkpoint switch
        self._session_gen = 0

        self.video_source = None
        self.cap = None
        # External frame source support (callable, iterator, pre-opened capture)
        self._external_frame_source = None
        self._cap_owned = False
        self.is_running = False
        self._is_video_file = False
        self._video_ended = False

        # ── Raw frame from reader (always latest, always smooth) ──
        self._raw_frame = None
        self._raw_lock = threading.Lock()
        self._raw_changed = threading.Event()

        # ── Frame offered to detector ──
        self._det_frame = None
        self._det_frame_idx = 0
        self._det_event = threading.Event()
        self._det_lock = threading.Lock()

        # ── Detection overlay data ──
        self._overlay = self._empty_overlay()
        self._overlay_frame = None   # frame that was used for this overlay (video sync)
        self._overlay_lock = threading.Lock()

        # ── Pre-encoded JPEG bytes (produced by compositor thread) ──
        self._jpeg_bytes = None
        self._jpeg_lock = threading.Lock()

        # ── Frame dimensions (set by reader, used by compositor for exit line) ──
        self._frame_width = 0
        self._frame_height = 0

        # ── Exit line Y position (set once by detector from first frame, never via overlay) ──
        self._exit_line_y = 0
        # ── Exit line as % from leading edge (survives sessions & rotation changes) ──
        self._exit_line_pct = 85
        # ── Exit line orientation: False = horizontal (y), True = vertical (x) ──
        self._exit_line_vertical = False
        # ── Exit line direction inverted: % measured from opposite edge ──
        # e.g. 85% normally = 85% from top; inverted = 85% from bottom (= 15% from top)
        self._exit_line_inverted = False
        # ── Exit line enabled flag (can be toggled via API, survives sessions) ──
        self._exit_line_enabled = True
        # ── Frame rotation steps (0,1,2,3 => 0°,90°,180°,270° CCW; survives sessions) ──
        self._rotation_steps = 0

        # ── Pause / resume ──
        self._paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # initially not paused

        # ── Video playback controls (video files only) ──
        self._playback_speed = 1.0
        self._video_pos_frames = 0
        self._video_total_frames = 0
        self._video_fps = 0
        self._seek_target = None  # frame number to seek to (atomic)

        # ── Raw mode (no detection overlay) ──
        self._raw_mode = False

        # ── Per-session tracking state ──
        self.frame_count = 0
        self.packages = {}
        self.total_packets = 0
        self.output_fifo = []
        self.packet_numbers = {}
        self.packets_crossed_line = set()

        # ── Stats for API ──
        self._stats_lock = threading.Lock()
        self.stats = self._empty_stats()

        # ── DB writer (Thread 4 — fully async, never blocks detection) ──
        self._db_writer     = DBWriter() if _DB_AVAILABLE else None
        self._db_session_id = None        # UUID of current session (None = no active session)
        self._stats_active  = False       # OFF by default — user activates via toggle button
        # Per-session NOK sub-counters
        self._nok_no_barcode = 0
        self._nok_no_date    = 0
        self._nok_both       = 0
        if self._db_writer:
            self._db_writer.start()

    # ─────────────────────────────────────────

    @staticmethod
    def _empty_overlay():
        return {
            'track_boxes': [],
            'barcode_boxes': [],
            'date_boxes': [],
            'exit_line_y': 0,
            'total_packets': 0,
            'fifo_str': '(empty)',
            'det_fps': 0,
            'det_ms': 0,
            'frame_idx': 0,
        }

    @staticmethod
    def _empty_stats():
        return {
            "video_fps": 0,
            "det_fps": 0,
            "inference_ms": 0,
            "total_packets": 0,
            "packages_ok": 0,
            "packages_nok": 0,
            "rotation_deg": 0,
            "fifo_queue": [],
            "is_running": False,
            "video_ended": False,
            "paused": False,
        }

    def _reset_session(self):
        self.packages = {}
        self.frame_count = 0
        self.total_packets = 0
        self.output_fifo = []
        self.packet_numbers = {}
        self.packets_crossed_line = set()
        self._video_ended = False
        self._raw_frame = None
        self._det_frame = None
        self._det_frame_idx = 0
        self._det_event.clear()
        self._raw_changed.clear()
        # Pause/seek reset
        self._paused = False
        self._pause_event.set()
        self._video_pos_frames = 0
        self._video_total_frames = 0
        self._video_fps = 0
        self._seek_target = None
        # Don't reset _playback_speed, _raw_mode — user preferences survive sessions
        # Reset exit line Y so detector recomputes it from CONFIG["exit_line_ratio"]
        self._exit_line_y = 0
        with self._jpeg_lock:
            self._jpeg_bytes = None
        with self._overlay_lock:
            self._overlay = self._empty_overlay()
        with self._stats_lock:
            self.stats = self._empty_stats()
            self.stats["rotation_deg"] = (self._rotation_steps % 4) * 90
        # Reset NOK sub-counters
        self._nok_no_barcode = 0
        self._nok_no_date    = 0
        self._nok_both       = 0
        # Reset anomaly detection per-track states
        self._ad_track_states = {}

    @staticmethod
    def _rotate_frame_ccw(frame, steps):
        steps = steps % 4
        if steps == 1:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if steps == 2:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if steps == 3:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def cycle_rotation_ccw(self):
        self._rotation_steps = (self._rotation_steps + 1) % 4
        deg = self._rotation_steps * 90
        with self._stats_lock:
            self.stats["rotation_deg"] = deg
        # Recompute _exit_line_y for the new orientation
        self._recompute_exit_line_y()
        print(f"[ROTATE] Input rotation set to {deg}° CCW")
        return deg

    def _recompute_exit_line_y(self):
        """Recompute pixel exit line position from _exit_line_pct + displayed frame dims.

        _exit_line_vertical = False → horizontal line, ref = displayed height.
        _exit_line_vertical = True  → vertical line,   ref = displayed width.
        After rotation 90°/270° the frame is transposed so height↔width swap.
        _exit_line_inverted = True  → % measured from the far edge instead of near edge
                                      (i.e. pixel = ref * (100 - pct) / 100).
        """
        steps = self._rotation_steps % 4
        transposed = steps in (1, 3)
        if self._exit_line_vertical:
            ref = self._frame_height if transposed else self._frame_width
        else:
            ref = self._frame_width if transposed else self._frame_height
        if ref > 0:
            effective_pct = (100 - self._exit_line_pct) if self._exit_line_inverted else self._exit_line_pct
            self._exit_line_y = int(ref * effective_pct / 100)

    # ═══════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════

    def start_processing(self, video_source):
        if self.is_running:
            print("[START] Restarting with new source...")
            self.is_running = False
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

        # Accept flexible video sources
        self.video_source = video_source
        self._external_frame_source = None
        self._cap_owned = False
        if hasattr(video_source, "read") and callable(getattr(video_source, "read")):
            self.cap = video_source
            self._cap_owned = False
        elif hasattr(video_source, "__iter__") and not isinstance(video_source, (str, bytes)):
            self._external_frame_source = iter(video_source)
            self.cap = None
        elif callable(video_source) and not isinstance(video_source, str):
            self._external_frame_source = video_source
            self.cap = None

        # Save current position BEFORE reset so we can resume from it
        _resume_frame = self._video_pos_frames if self._is_video_file else 0
        if self._video_ended:
            _resume_frame = 0
        self._reset_session()
        self._is_video_file = isinstance(video_source, str) and any(
            video_source.lower().endswith(ext)
            for ext in VIDEO_EXTENSIONS
        )
        if self._is_video_file and _resume_frame > 0:
            self._seek_target = _resume_frame

        # Reset built-in tracker state for fresh session
        if hasattr(self.model, 'predictor') and self.model.predictor is not None:
            self.model.predictor.trackers = []
            self.model.predictor = None

        self._session_gen += 1
        my_gen = self._session_gen
        self.is_running = True

        threading.Thread(target=self._reader_loop,     args=(my_gen,), daemon=True, name="VideoReader").start()
        threading.Thread(target=self._detection_loop,  daemon=True, name="YOLODetector").start()
        threading.Thread(target=self._compositor_loop, daemon=True, name="Compositor").start()

        mode = "video_file" if self._is_video_file else "live"
        return {"status": "started", "source": video_source, "mode": mode}

    def stop_processing(self, preserve_pause: bool = False):
        # Session lifecycle is managed by the toggle button, not by stop.
        # If recording is active when stop is called, leave session open
        # so it continues when processing resumes.
        # By default stopping will clear the paused state, but callers can
        # request the paused state to be preserved (useful for checkpoint
        # switches so a paused UI remains paused after reload).
        self.is_running = False
        if not preserve_pause:
            self._paused = False
            self._pause_event.set()
        else:
            # Keep pause flag as-is. Ensure the pause event matches the flag
            # (cleared when paused, set when not paused).
            if getattr(self, '_paused', False):
                self._pause_event.clear()
            else:
                self._pause_event.set()
        time.sleep(0.5)
        if self.cap:
            try:
                if self._cap_owned:
                    self.cap.release()
            except Exception:
                pass
            self.cap = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        import gc
        gc.collect()
        with self._stats_lock:
            self.stats["is_running"] = False
            self.stats["paused"] = False
        return {"status": "stopped"}

    def pause_processing(self):
        """Pause playback. Threads stay alive, cap stays open."""
        if not self.is_running:
            return {"error": "Not running"}
        self._paused = True
        self._pause_event.clear()
        with self._stats_lock:
            self.stats["paused"] = True
        print("[PAUSE] Video paused")
        return {"status": "paused", "frame": self._video_pos_frames}

    def resume_processing(self):
        """Resume from pause."""
        # If processing is not running, try to start it using the last
        # known `video_source` so the UI Resume button can be used after
        # a Stop + checkpoint change (user expects to continue where they left off).
        if not self.is_running:
            if getattr(self, 'video_source', None):
                self.start_processing(self.video_source)
            else:
                return {"error": "Not running"}
        # Clear paused flag / event to continue processing
        self._paused = False
        self._pause_event.set()
        with self._stats_lock:
            self.stats["paused"] = False
        print("[RESUME] Video resumed")
        return {"status": "resumed"}

    def seek_video(self, position_pct):
        """Queue a seek to position_pct% (0-100). Reader thread applies it."""
        if not self._is_video_file or self._video_total_frames <= 0:
            return {"error": "Seek only available for video files"}
        pct = max(0, min(100, float(position_pct)))
        target = int(self._video_total_frames * pct / 100)
        self._seek_target = target
        return {"status": "seeking", "target_frame": target, "position_pct": pct}

    def set_playback_speed(self, speed):
        """Set playback speed multiplier (0.1 – 4.0)."""
        self._playback_speed = max(0.1, min(4.0, float(speed)))
        print(f"[SPEED] Playback speed set to {self._playback_speed:.2f}x")
        return {"speed": self._playback_speed}

    @staticmethod
    def _compute_iou(box1, box2):
        """Compute IoU between two (x1,y1,x2,y2) boxes."""
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    # ═══════════════════════════════════════════
    # CHECKPOINT SWITCHING  (unloads + reloads)
    # ═══════════════════════════════════════════

    def switch_checkpoint(self, checkpoint: dict):
        """
        Unload the current model from VRAM, load the new checkpoint,
        and restart processing on the same source if it was running.
        Returns a status dict.
        """
        from ultralytics import YOLO
        import torch, gc

        was_running = self.is_running
        prev_source = self.video_source

        # 1. Stop current processing
        if was_running:
            print(f"[SWITCH] Stopping current processing...")
            # Preserve paused state so a user-paused session remains paused
            # after switching checkpoints.
            self.stop_processing(preserve_pause=True)

        # 2. Unload models from VRAM
        if self.model is not None:
            print(f"[SWITCH] Unloading primary model from VRAM...")
            del self.model
            self.model = None
        if self.secondary_model is not None:
            print(f"[SWITCH] Unloading secondary date model from VRAM...")
            del self.secondary_model
            self.secondary_model = None
            self._secondary_date_id = None
            self._use_secondary_date = False
        # Unload EfficientAD models
        for attr in ('_ad_teacher', '_ad_student', '_ad_autoencoder'):
            if getattr(self, attr, None) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        self._ad_mean = None
        self._ad_std = None
        self._ad_quantiles = None
        self._ad_transform = None
        self._ad_track_states = {}
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        print(f"[SWITCH] VRAM freed.")

        # 3. Load new model
        print(f"[SWITCH] Loading checkpoint: {checkpoint['label']} ({checkpoint['path']})")
        from tracking_config import DEVICE
        self.model = YOLO(checkpoint["path"])
        try:
            self.model.to(DEVICE)
        except Exception:
            pass
        # attempt FP16 on CUDA (skip for anomaly/segmentation models)
        if self.mode != "anomaly":
            try:
                if str(DEVICE).lower().startswith("cuda") and torch.cuda.is_available():
                    try:
                        getattr(self.model, "model", None).half()
                        print("[SWITCH] Converted model to FP16 (half precision).")
                    except Exception as e:
                        print(f"[SWITCH] FP16 conversion failed: {e}")
            except Exception:
                pass
        names = self.model.names

        # 4. Resolve class IDs
        pkg_cls = checkpoint.get("package_class")
        bar_cls = checkpoint.get("barcode_class")
        self.package_id = next((k for k, v in names.items() if v == pkg_cls), None) if pkg_cls else None
        self.barcode_id = next((k for k, v in names.items() if v == bar_cls), None) if bar_cls else None
        date_cls = checkpoint.get("date_class")
        self.date_id = next((k for k, v in names.items() if v == date_cls), None) if date_cls else None
        self.mode = checkpoint.get("mode", "tracking")
        self.current_checkpoint = checkpoint

        # Anomaly mode doesn't use an exit line — disable it automatically
        if self.mode == "anomaly":
            self._exit_line_enabled = False

        # Ensure each checkpoint has explicit per-class thresholds so
        # switching to a model that doesn't specify them won't break
        # detection. Prefer values defined on the checkpoint, then fall
        # back to global `CONFIG` entries. If those are absent, use
        # conservative numeric fallbacks.
        try:
            from tracking_config import CONFIG as _GLOBAL_CONFIG
        except Exception:
            _GLOBAL_CONFIG = {}

        missing = []
        for k, fallback in (('conf_paquet', 0.45), ('conf_barcode', 0.45), ('conf_date', 0.30)):
            if k not in self.current_checkpoint or self.current_checkpoint.get(k) is None:
                if k in _GLOBAL_CONFIG:
                    self.current_checkpoint[k] = _GLOBAL_CONFIG[k]
                else:
                    self.current_checkpoint[k] = fallback
                missing.append(k)
        if missing:
            print(f"[SWITCH] Populated missing thresholds for checkpoint: {', '.join(missing)}")

        print(f"[SWITCH] Loaded | mode={self.mode} | "
              f"package_id={self.package_id} barcode_id={self.barcode_id} date_id={self.date_id}")

        # 4a. Load secondary date model if configured
        sec_path = checkpoint.get("secondary_date_model_path")
        sec_cls  = checkpoint.get("secondary_date_class")
        if sec_path and self.mode == "tracking":
            print(f"[SWITCH] Loading secondary date model: {sec_path}")
            self.secondary_model = YOLO(sec_path)
            try:
                self.secondary_model.to(DEVICE)
            except Exception:
                pass
            try:
                if str(DEVICE).lower().startswith("cuda") and torch.cuda.is_available():
                    getattr(self.secondary_model, "model", None).half()
                    print("[SWITCH] Secondary model converted to FP16.")
            except Exception:
                pass
            sec_names = self.secondary_model.names
            self._secondary_date_id = next(
                (k for k, v in sec_names.items() if v == sec_cls), None
            ) if sec_cls else None
            self._use_secondary_date = self._secondary_date_id is not None
            # Warmup secondary model
            try:
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self.secondary_model(dummy, imgsz=CONFIG["imgsz"], verbose=False)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[SWITCH] Secondary model warmup done | date_id={self._secondary_date_id}")
            except Exception as e:
                print(f"[SWITCH] Secondary warmup failed (non-fatal): {e}")
        else:
            self.secondary_model = None
            self._secondary_date_id = None
            self._use_secondary_date = False

        # 4b. Load EfficientAD models if anomaly mode
        if self.mode == "anomaly":
            self._load_ad_models(checkpoint, DEVICE)
        else:
            self._ad_teacher = None
            self._ad_student = None
            self._ad_autoencoder = None
            self._ad_transform = None
            self._ad_track_states = {}

        # 4c. Apply default rotation for this checkpoint
        default_rot = checkpoint.get("default_rotation", 0)
        if default_rot != self._rotation_steps:
            self._rotation_steps = default_rot % 4
            deg = self._rotation_steps * 90
            with self._stats_lock:
                self.stats["rotation_deg"] = deg
            self._recompute_exit_line_y()
            print(f"[SWITCH] Rotation set to {deg}° CCW (checkpoint default)")

        # 5. Warm up
        try:
            from tracking_config import CONFIG
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, imgsz=CONFIG["imgsz"], verbose=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[SWITCH] Warmup done.")
        except Exception as e:
            print(f"[SWITCH] Warmup failed (non-fatal): {e}")

        return {
            "status": "switched",
            "checkpoint_id": checkpoint["id"],
            "label": checkpoint["label"],
            "mode": self.mode,
            "was_running": was_running,
            "prev_source": prev_source,
        }

    # ═══════════════════════════════════════════
    # ANOMALY DETECTION HELPERS
    # ═══════════════════════════════════════════

    def _load_ad_models(self, checkpoint, device):
        """Load EfficientAD teacher/student/autoencoder for anomaly mode."""
        import torch
        from torchvision import transforms
        from anomaly_on_video import get_ad_constants

        ad_teacher_path = checkpoint.get("ad_teacher", "teacher_best.pth")
        ad_student_path = checkpoint.get("ad_student", "student_best.pth")
        ad_ae_path = checkpoint.get("ad_autoencoder", "autoencoder_best.pth")
        ad_imgsz = checkpoint.get("ad_imgsz", 256)

        print(f"[SWITCH] Loading EfficientAD models...")
        self._ad_teacher = torch.load(ad_teacher_path, map_location=device, weights_only=False).eval()
        self._ad_student = torch.load(ad_student_path, map_location=device, weights_only=False).eval()
        self._ad_autoencoder = torch.load(ad_ae_path, map_location=device, weights_only=False).eval()

        self._ad_mean, self._ad_std, self._ad_quantiles = get_ad_constants(device)

        self._ad_transform = transforms.Compose([
            transforms.Resize((ad_imgsz, ad_imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._ad_track_states = {}

        # Warmup EfficientAD
        try:
            dummy = torch.zeros(1, 3, ad_imgsz, ad_imgsz, device=device)
            from efficientad import predict as effpredict
            effpredict(
                image=dummy, teacher=self._ad_teacher, student=self._ad_student,
                autoencoder=self._ad_autoencoder, teacher_mean=self._ad_mean,
                teacher_std=self._ad_std,
                q_st_start=self._ad_quantiles['q_st_start'],
                q_st_end=self._ad_quantiles['q_st_end'],
                q_ae_start=self._ad_quantiles['q_ae_start'],
                q_ae_end=self._ad_quantiles['q_ae_end']
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[SWITCH] EfficientAD warmup done.")
        except Exception as e:
            print(f"[SWITCH] EfficientAD warmup failed (non-fatal): {e}")

    def _ad_crop_and_mask(self, frame, mask_raw, checkpoint):
        """Crop object using segmentation mask, blackout background, letterbox."""
        h_f, w_f = frame.shape[:2]
        mask_full = cv2.resize(mask_raw, (w_f, h_f), interpolation=cv2.INTER_LINEAR)
        _, mask_full = cv2.threshold(mask_full, 0.5, 1, cv2.THRESH_BINARY)

        rows, cols = np.where(mask_full > 0)
        if len(rows) == 0:
            return None
        x1, y1, x2, y2 = np.min(cols), np.min(rows), np.max(cols), np.max(rows)

        w_box, h_box = x2 - x1, y2 - y1
        margin = checkpoint.get("ad_margin_pct", 0.1)
        m_x, m_y = int(w_box * margin), int(h_box * margin)

        cx1, cy1 = max(0, x1 - m_x), max(0, y1 - m_y)
        cx2, cy2 = min(w_f, x2 + m_x), min(h_f, y2 + m_y)

        img_crop = frame[cy1:cy2, cx1:cx2].copy()
        mask_crop = mask_full[cy1:cy2, cx1:cx2].copy()

        erosion_size = checkpoint.get("ad_erosion_size", 3)
        if erosion_size > 0:
            mask_blurred = cv2.GaussianBlur(mask_crop, (5, 5), 0)
            _, mask_final = cv2.threshold(mask_blurred, 0.5, 1, cv2.THRESH_BINARY)
            mask_final = mask_final.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
            mask_final = cv2.erode(mask_final, kernel, iterations=1)
        else:
            mask_final = mask_crop.astype(np.uint8)

        img_crop[mask_final == 0] = 0
        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        return letterbox_image(img_rgb)

    def _ad_detect_anomaly(self, img_crop_np):
        """Run EfficientAD on a preprocessed crop. Returns (is_defective, score)."""
        import torch
        from PIL import Image
        from efficientad import predict as effpredict

        pil_img = Image.fromarray(img_crop_np)
        orig_w, orig_h = pil_img.size
        img_tensor = self._ad_transform(pil_img).unsqueeze(0)
        device = next(self._ad_teacher.parameters()).device
        img_tensor = img_tensor.to(device)

        map_combined, _, _ = effpredict(
            image=img_tensor, teacher=self._ad_teacher, student=self._ad_student,
            autoencoder=self._ad_autoencoder, teacher_mean=self._ad_mean,
            teacher_std=self._ad_std,
            q_st_start=self._ad_quantiles['q_st_start'],
            q_st_end=self._ad_quantiles['q_st_end'],
            q_ae_start=self._ad_quantiles['q_ae_start'],
            q_ae_end=self._ad_quantiles['q_ae_end']
        )

        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_h, orig_w), mode='bilinear')
        score = map_combined[0, 0].cpu().numpy().max()

        thresh = (self.current_checkpoint or {}).get("ad_thresh", 5000.0)
        return score > thresh, float(score)

    @staticmethod
    def _ad_final_decision(results, strategy="MAJORITY"):
        """Aggregate per-frame anomaly results into a final decision."""
        if not results:
            return False
        if strategy == "OR":
            return any(results)
        # MAJORITY
        return sum(results) > (len(results) / 2)

    # ═══════════════════════════════════════════
    # THREAD 1: VIDEO READER (smooth, native FPS)
    # ═══════════════════════════════════════════

    def _reader_loop(self, session_gen: int):
        """Read frames at native FPS. NEVER waits for YOLO. Always smooth."""
        try:
            src = self.video_source
            ext_src = self._external_frame_source

            if isinstance(src, str) and src.startswith("rtsp://") and ext_src is None:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                self._cap_owned = True
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif self._is_video_file and ext_src is None:
                if not Path(src).exists():
                    print(f"[READER] ERROR: File not found: {src}")
                    self.is_running = False
                    return
                self.cap = cv2.VideoCapture(src)
                self._cap_owned = True
            elif ext_src is None and self.cap is None:
                import platform
                if isinstance(src, str) and src.isdigit():
                    src = int(src)
                if platform.system() == "Windows":
                    self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
                self._cap_owned = True
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Verify capture is opened (skip check for external callable/iterator)
            if self.cap is not None and not self.cap.isOpened():
                print(f"[READER] ERROR: Cannot open source: {src}")
                self.is_running = False
                return

            # Determine FPS/size from capture if available
            if self.cap is not None:
                raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
                fps = raw_fps if raw_fps and raw_fps > 0 else CAMERA_FPS
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                fps = CAMERA_FPS
                w = CAMERA_WIDTH
                h = CAMERA_HEIGHT
            frame_time = 1.0 / fps

            with self._stats_lock:
                self.stats["video_fps"] = round(fps, 1)
                self.stats["is_running"] = True

            self._frame_width = w
            self._frame_height = h
            self._video_fps = fps

            # For video files, get total frame count for timeline
            if self._is_video_file and self.cap is not None:
                self._video_total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                dur = self._video_total_frames / fps if fps > 0 else 0
                print(f"[READER] Opened: {w}x{h} @ {fps:.0f}fps | "
                      f"Video file | {self._video_total_frames} frames | {dur:.1f}s")
            else:
                self._video_total_frames = 0
                print(f"[READER] Opened: {w}x{h} @ {fps:.0f}fps | Live camera")

            while self.is_running:
                t0 = time.time()

                # ── Handle seek request (works even while paused) ──
                _seek = self._seek_target
                if _seek is not None:
                    self._seek_target = None
                    if self.cap is not None:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, _seek)
                    self._video_pos_frames = _seek
                    # Reset tracker state on seek
                    self.packages.clear()
                    self.packets_crossed_line.clear()
                    self.total_packets = 0
                    self.output_fifo.clear()
                    self.packet_numbers.clear()
                    if hasattr(self.model, 'predictor') and self.model.predictor is not None:
                        self.model.predictor.trackers = []
                        self.model.predictor = None
                    with self._stats_lock:
                        self.stats["total_packets"] = 0
                        self.stats["packages_ok"] = 0
                        self.stats["packages_nok"] = 0
                        self.stats["fifo_queue"] = []

                # ── Pause: sleep and re-check (seek above still works) ──
                if self._paused and _seek is None:
                    time.sleep(0.05)
                    continue

                # ── Read frame ──
                if ext_src is not None and not (hasattr(ext_src, "read") and callable(getattr(ext_src, "read"))):
                    frm = None
                    try:
                        if callable(ext_src):
                            frm = ext_src()
                        else:
                            frm = next(ext_src)
                    except StopIteration:
                        ret, frame = False, None
                    except Exception as e:
                        print(f"[READER] External source error: {e}")
                        ret, frame = False, None
                    else:
                        if isinstance(frm, tuple) and len(frm) >= 2:
                            ret, frame = frm[0], frm[1]
                        else:
                            frame = frm
                            ret = frame is not None
                else:
                    ret, frame = self.cap.read()

                if not ret:
                    if self._is_video_file:
                        print("[READER] Video file ended")
                        self._video_ended = True
                        with self._stats_lock:
                            self.stats["video_ended"] = True
                    else:
                        print("[READER] Source ended or returned no frame")
                    break

                self.frame_count += 1

                # Track video position for timeline
                if self._is_video_file and self.cap is not None:
                    self._video_pos_frames = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                # Optional live rotation
                rot_steps = self._rotation_steps % 4
                if rot_steps:
                    frame = self._rotate_frame_ccw(frame, rot_steps)

                # Store raw frame for streaming
                with self._raw_lock:
                    self._raw_frame = frame
                self._raw_changed.set()

                # Send 1 frame every N to detector
                if self.frame_count % DETECTOR_FRAME_SKIP == 0:
                    with self._det_lock:
                        self._det_frame = frame.copy()
                        self._det_frame_idx = self.frame_count
                    self._det_event.set()

                # If we just processed a seek frame while paused, go back to waiting
                if self._paused:
                    continue

                # For video files: respect native FPS (speed-adjusted)
                if self._is_video_file:
                    speed = max(0.1, self._playback_speed)
                    elapsed = time.time() - t0
                    sleep = (frame_time / speed) - elapsed
                    if sleep > 0:
                        time.sleep(sleep)

        except Exception as e:
            print(f"[READER] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            time.sleep(0.1)
            if self.cap and self._cap_owned:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
            if self._session_gen == session_gen:
                self.is_running = False
                with self._stats_lock:
                    self.stats["is_running"] = False
            print(f"[READER] Stopped (gen={session_gen})")

    # ═══════════════════════════════════════════
    # THREAD 2: YOLO + BYTETRACK (parallel)
    # ═══════════════════════════════════════════

    def _detection_loop(self):
        """Run YOLO + ByteTrack in parallel. Updates overlay & stats independently."""
        try:
            print("[DETECTOR] Waiting for first frame...")
            for attempt in range(200):
                if not self.is_running:
                    return
                if self._det_event.wait(timeout=0.1):
                    break
            else:
                print("[DETECTOR] TIMEOUT: No frame after 20s, exiting")
                return

            with self._det_lock:
                first_frame = self._det_frame
            if first_frame is None:
                print("[DETECTOR] No frame received, exiting")
                return

            height, width = first_frame.shape[:2]
            # Compute from config ratio
            self._exit_line_pct = round((1.0 - CONFIG["exit_line_ratio"]) * 100)
            self._recompute_exit_line_y()

            with self._overlay_lock:
                self._overlay['exit_line_y'] = self._exit_line_y

            steps = self._rotation_steps % 4
            orientation = 'horizontal' if steps in (0, 2) else 'vertical'
            print(f"[DETECTOR] Started | {width}x{height} | Exit={self._exit_line_y}px "
                  f"({self._exit_line_pct}% | {orientation})")

            last_processed_idx = 0

            while self.is_running:
                if not self._det_event.wait(timeout=1.0):
                    if not self.is_running:
                        break
                    continue
                self._det_event.clear()

                with self._det_lock:
                    frame = self._det_frame
                    frame_idx = self._det_frame_idx

                if frame is None or frame_idx <= last_processed_idx:
                    continue

                last_processed_idx = frame_idx
                t_start = time.time()

                # ── YOLO Inference ──
                try:
                    # Build list of per-class thresholds from the active
                    # checkpoint (populated during `switch_checkpoint`).
                    # Use ALL thresholds regardless of which class IDs are
                    # resolved, so even checkpoints where class names are
                    # None (e.g. "date" mode) still get a sane conf floor.
                    cp = self.current_checkpoint or {}
                    conf_list = [
                        v for k in ("conf_paquet", "conf_barcode", "conf_date")
                        if (v := cp.get(k)) is not None
                    ]
                    conf_min = min(conf_list) if conf_list else CONFIG.get("conf_date", 0.25)

                    # Per-checkpoint overrides for imgsz / conf
                    yolo_imgsz = cp.get("yolo_imgsz", CONFIG["imgsz"])
                    yolo_conf = cp.get("yolo_conf", conf_min)

                    if self.mode == "tracking":
                        results = self.model.track(
                            frame,
                            half=True,
                            conf=conf_min,
                            imgsz=CONFIG["imgsz"],
                            verbose=False,
                            persist=True,
                            tracker=TRACKER_YAML_PATH,
                        )[0]
                    elif self.mode == "anomaly":
                        results = self.model.track(
                            frame,
                            half=False,
                            conf=yolo_conf,
                            imgsz=yolo_imgsz,
                            verbose=False,
                            persist=True,
                            tracker=TRACKER_YAML_PATH,
                            retina_masks=True,
                        )[0]
                    else:
                        results = self.model(
                            frame,
                            conf=conf_min,
                            imgsz=CONFIG["imgsz"],
                            verbose=False
                        )[0]
                except Exception as yolo_err:
                    print(f"[DETECTOR] YOLO inference error: {yolo_err}")
                    continue

                # ── DATE DETECTION mode ──
                if self.mode == "date":
                    date_boxes = []
                    if results.boxes is not None:
                        for b in results.boxes:
                            cls  = int(b.cls)
                            conf = float(b.conf)
                            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                            label = self.model.names.get(cls, str(cls))
                            date_boxes.append((int(x1), int(y1), int(x2), int(y2),
                                               f"{label} {conf:.2f}", (0, 200, 255)))

                    det_ms  = (time.time() - t_start) * 1000
                    det_fps = 1000 / det_ms if det_ms > 0 else 0

                    with self._overlay_lock:
                        self._overlay = {
                            'track_boxes':   date_boxes,
                            'barcode_boxes': [],
                            'date_boxes':    [],
                            'exit_line_y':   self._exit_line_y,
                            'total_packets': 0,
                            'fifo_str':      '(date mode)',
                            'det_fps':       det_fps,
                            'det_ms':        det_ms,
                            'frame_idx':     frame_idx,
                        }
                        self._overlay_frame = frame
                    with self._stats_lock:
                        self.stats.update({
                            "det_fps":       round(det_fps, 1),
                            "inference_ms":  round(det_ms, 1),
                        })
                    continue

                # ── ANOMALY DETECTION mode ──
                if self.mode == "anomaly":
                    cp = self.current_checkpoint or {}
                    zone_start_pct = cp.get("zone_start_pct", 0.20)
                    zone_end_pct = cp.get("zone_end_pct", 0.60)
                    ad_strategy = cp.get("ad_strategy", "MAJORITY")
                    h_f, w_f = frame.shape[:2]
                    zone_start_px = int(w_f * zone_start_pct)
                    zone_end_px = int(w_f * zone_end_pct)

                    track_boxes = []
                    ad_zone_lines = (zone_start_px, zone_end_px)

                    has_masks = results.masks is not None
                    has_ids = results.boxes is not None and results.boxes.id is not None

                    if has_masks and has_ids:
                        masks = results.masks.data.cpu().numpy()
                        boxes = results.boxes.xyxy.cpu().numpy()
                        track_ids = results.boxes.id.int().cpu().tolist()

                        for i, tid in enumerate(track_ids):
                            if tid not in self._ad_track_states:
                                self._ad_track_states[tid] = {'results': [], 'decision': None}
                            tstate = self._ad_track_states[tid]

                            x1, y1, x2, y2 = map(int, boxes[i])
                            center_x = (x1 + x2) // 2

                            label = "WAITING"
                            color = (200, 200, 200)  # grey

                            # Before zone: entering
                            if center_x > zone_end_px:
                                label = f"T{tid} ENTERING"
                                color = (200, 200, 200)

                            # Inside zone: scan with EfficientAD
                            elif zone_start_px <= center_x <= zone_end_px and tstate['decision'] is None:
                                img_crop = self._ad_crop_and_mask(frame, masks[i], cp)
                                if img_crop is not None:
                                    try:
                                        is_def, score = self._ad_detect_anomaly(img_crop)
                                        tstate['results'].append(is_def)
                                        label = f"T{tid} SCANNING"
                                        color = (0, 255, 255)  # cyan
                                    except Exception as ad_err:
                                        print(f"[AD] Error on track {tid}: {ad_err}")
                                        label = f"T{tid} AD-ERR"
                                        color = (0, 165, 255)  # orange
                                else:
                                    label = f"T{tid} SCANNING"
                                    color = (0, 255, 255)

                            # Past zone: lock decision
                            else:
                                if tstate['decision'] is None:
                                    tstate['decision'] = self._ad_final_decision(
                                        tstate['results'], strategy=ad_strategy)
                                is_def = tstate['decision']

                                if tid not in self.packets_crossed_line:
                                    self.packets_crossed_line.add(tid)
                                    self.total_packets += 1
                                    self.packet_numbers[tid] = self.total_packets
                                    final = "NOK" if is_def else "OK"
                                    self.output_fifo.append(final)
                                    print(f"[AD] Packet #{self.total_packets} -> {final} "
                                          f"(scans={len(tstate['results'])})")

                                    # ── Save image locally for NOK packets only ──
                                    if final == "NOK":
                                        try:
                                            # Create folder structure: images/{session_id}/anomaly/
                                            session_dir = os.path.join(os.path.dirname(__file__), "images", str(self._db_session_id))
                                            defect_dir = os.path.join(session_dir, "anomaly")
                                            os.makedirs(defect_dir, exist_ok=True)

                                            # Save image as {packet_number}.jpg
                                            image_filename = f"{self.total_packets}.jpg"
                                            image_path = os.path.join(defect_dir, image_filename)
                                            cv2.imwrite(image_path, frame)

                                            # Record image in database
                                            relative_path = f"{self._db_session_id}/anomaly/{image_filename}"
                                            if self._db_writer:
                                                self._db_writer.record_image(self._db_session_id, self.total_packets, "anomaly", relative_path)

                                            print(f"[IMAGE] Saved NOK packet {self.total_packets} to anomaly folder")
                                        except Exception as img_err:
                                            print(f"[IMAGE] Error saving image: {img_err}")
                                if is_def:
                                    label = f"#{pkt_num} DEFECTIVE" if pkt_num else f"T{tid} DEFECTIVE"
                                    color = (0, 0, 255)  # red
                                else:
                                    label = f"#{pkt_num} GOOD" if pkt_num else f"T{tid} GOOD"
                                    color = (0, 255, 0)  # green

                            track_boxes.append((x1, y1, x2, y2, label, color))

                    det_ms = (time.time() - t_start) * 1000
                    det_fps = 1000 / det_ms if det_ms > 0 else 0

                    ok = self.output_fifo.count("OK")
                    nok = self.output_fifo.count("NOK")

                    with self._overlay_lock:
                        self._overlay = {
                            'track_boxes': track_boxes,
                            'barcode_boxes': [],
                            'date_boxes': [],
                            'exit_line_y': self._exit_line_y,
                            'total_packets': self.total_packets,
                            'fifo_str': '(anomaly mode)',
                            'det_fps': det_fps,
                            'det_ms': det_ms,
                            'frame_idx': frame_idx,
                            'ad_zone_lines': ad_zone_lines,
                        }
                        self._overlay_frame = frame
                    with self._stats_lock:
                        self.stats.update({
                            "det_fps": round(det_fps, 1),
                            "inference_ms": round(det_ms, 1),
                            "total_packets": self.total_packets,
                            "packages_ok": ok,
                            "packages_nok": nok,
                            "fifo_queue": list(self.output_fifo[-10:]),
                        })
                    continue

                # ── Extract tracked packages, barcode and date detections ──
                tracks = []
                barcode_dets = []
                date_dets = []

                if results.boxes is not None:
                    box_ids = results.boxes.id
                    for i, b in enumerate(results.boxes):
                        cls = int(b.cls)
                        conf = float(b.conf)
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                        if self.package_id is not None and cls == self.package_id and conf >= self.current_checkpoint.get("conf_paquet", CONFIG.get("conf_paquet")):
                            tid = int(box_ids[i]) if box_ids is not None else -1
                            if tid >= 0:
                                tracks.append([int(x1), int(y1), int(x2), int(y2), tid])
                        elif self.barcode_id is not None and cls == self.barcode_id and conf >= self.current_checkpoint.get("conf_barcode", CONFIG.get("conf_barcode")):
                            barcode_dets.append([x1, y1, x2, y2, conf])
                        elif self.date_id is not None and cls == self.date_id and conf >= self.current_checkpoint.get("conf_date", CONFIG.get("conf_date")):
                            date_dets.append([int(x1), int(y1), int(x2), int(y2), conf])

                # ── Secondary date model inference (if active) ──
                secondary_date_dets = []
                if self._use_secondary_date and self.secondary_model is not None:
                    try:
                        sec_conf = self.current_checkpoint.get("conf_date", CONFIG.get("conf_date", 0.30))
                        sec_results = self.secondary_model(
                            frame,
                            conf=sec_conf,
                            imgsz=CONFIG["imgsz"],
                            verbose=False
                        )[0]
                        if sec_results.boxes is not None:
                            for b in sec_results.boxes:
                                cls = int(b.cls)
                                conf_val = float(b.conf)
                                if cls == self._secondary_date_id:
                                    sx1, sy1, sx2, sy2 = b.xyxy[0].cpu().numpy()
                                    secondary_date_dets.append([int(sx1), int(sy1), int(sx2), int(sy2), conf_val])
                    except Exception as sec_err:
                        print(f"[DETECTOR] Secondary date model error: {sec_err}")

                # Merge both sources then deduplicate via IoU (NMS-like)
                # so the same physical date label doesn't appear twice on screen.
                _merged = date_dets + secondary_date_dets
                all_date_dets = []
                for cand in sorted(_merged, key=lambda d: d[4], reverse=True):
                    cx1, cy1, cx2, cy2 = cand[0], cand[1], cand[2], cand[3]
                    duplicate = False
                    for kept in all_date_dets:
                        kx1, ky1, kx2, ky2 = kept[0], kept[1], kept[2], kept[3]
                        ix1, iy1 = max(cx1, kx1), max(cy1, ky1)
                        ix2, iy2 = min(cx2, kx2), min(cy2, ky2)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        union = (cx2 - cx1) * (cy2 - cy1) + (kx2 - kx1) * (ky2 - ky1) - inter
                        if union > 0 and inter / union > 0.4:
                            duplicate = True
                            break
                    if not duplicate:
                        all_date_dets.append(cand)

                # ── Per-track processing ──
                track_boxes = []

                for t in tracks:
                    x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])

                    if tid not in self.packages:
                        inherited_barcode = False
                        inherited_date = False
                        new_bbox = (x1, y1, x2, y2)
                        for etid, epkg in self.packages.items():
                            if epkg.get("prev_bbox") and self._compute_iou(new_bbox, epkg["prev_bbox"]) > 0.3:
                                if epkg.get("barcode_detected") and not inherited_barcode:
                                    inherited_barcode = True
                                    print(f"[DET] Track {tid} inherited barcode from Track {etid} (IoU match)")
                                if epkg.get("date_detected") and not inherited_date:
                                    inherited_date = True
                                    print(f"[DET] Track {tid} inherited date from Track {etid} (IoU match)")
                                if inherited_barcode and inherited_date:
                                    break
                        self.packages[tid] = {
                            "barcode_detected": inherited_barcode,
                            "date_detected": inherited_date,
                            "decision_locked": False,
                            "final_decision": None,
                            "prev_bbox": None,
                            "prev_area": None,
                            "frames_tracked": 0,
                            "first_frame": frame_idx,
                            "pre_line_seen": False,
                        }

                    pkg = self.packages[tid]
                    pkg["frames_tracked"] += 1
                    bbox = (x1, y1, x2, y2)

                    ca, _ = calculate_bbox_metrics(x1, y1, x2, y2)
                    pkg["prev_bbox"] = bbox
                    pkg["prev_area"] = ca

                    # ── Barcode association ──
                    if not pkg["barcode_detected"]:
                        for bx1, by1, bx2, by2, bc in barcode_dets:
                            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                            if x1 <= cx <= x2 and y1 <= cy <= y2:
                                pkg["barcode_detected"] = True
                                print(f"[DET] Barcode on Track {tid} conf={bc:.3f}")
                                break

                    # ── Date association (from primary + secondary model) ──
                    if not pkg.get("date_detected"):
                        for dx1, dy1, dx2, dy2, dc in all_date_dets:
                            cx, cy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
                            if x1 <= cx <= x2 and y1 <= cy <= y2:
                                pkg["date_detected"] = True
                                src = "secondary" if [dx1, dy1, dx2, dy2, dc] in secondary_date_dets else "primary"
                                print(f"[DET] Date on Track {tid} conf={dc:.3f} (from {src} model)")
                                break

                    # ── Visualization ──
                    # Determine OK/NOK based on which validations are required.
                    # When secondary date model is active: need BOTH barcode AND date.
                    # Otherwise: only barcode is required (original behaviour).
                    if pkg["decision_locked"]:
                        color = (255, 165, 0)
                        status = pkg["final_decision"]
                    else:
                        has_barcode = pkg["barcode_detected"]
                        has_date = pkg.get("date_detected", False)
                        if self._use_secondary_date:
                            # Dual validation: barcode + date
                            if has_barcode and has_date:
                                color = (0, 255, 0)    # green = both found
                                status = "OK"
                            elif has_barcode or has_date:
                                color = (0, 165, 255)  # orange = partial
                                what = "BC" if has_barcode else "DT"
                                status = f"NOK({what})"
                            else:
                                color = (0, 0, 255)    # red = nothing
                                status = "NOK"
                        else:
                            # Original: barcode only
                            if has_barcode:
                                color = (0, 255, 0)
                                status = "OK"
                            else:
                                color = (0, 0, 255)
                                status = "NOK"

                    if tid in self.packet_numbers:
                        lbl = f"#{self.packet_numbers[tid]} {status}"
                    else:
                        lbl = f"T{tid}|{status}"

                    track_boxes.append((x1, y1, x2, y2, lbl, color))

                # ── Exit line crossing (horizontal: y2; vertical: x2) ──
                if self._exit_line_enabled:
                    current_exit = self._exit_line_y
                    line_is_vert = self._exit_line_vertical
                    exit_pct = self._exit_line_pct  # e.g. 85 → line in the far portion
                    for t in tracks:
                        x1, y1, x2, y2, tid = map(int, t[:5])
                        if tid not in self.packages:
                            continue
                        pkg = self.packages[tid]

                        # ── Track whether the packet has been seen on the near side ──
                        # "near side" = the side from which the packet approaches.
                        # Not inverted (effective_pct > 50): line is in the far half,
                        #   packets travel toward larger coords → leading edge = x1/y1 (min).
                        #   pre_line_seen once x1/y1 < exit (still approaching).
                        #   crossed once x2/y2 >= exit (max edge clears the line).
                        # Inverted (effective_pct <= 50): line is in the near half,
                        #   packets travel toward smaller coords (e.g. right→left) →
                        #   leading edge = x1/y1 (min coord = front of travel).
                        #   pre_line_seen once x1/y1 > exit (still on approach side).
                        #   crossed once x1/y1 <= exit (front edge clears the line).
                        if not pkg["pre_line_seen"]:
                            effective_pct = (100 - exit_pct) if self._exit_line_inverted else exit_pct
                            if line_is_vert:
                                near_check = (x1 < current_exit) if effective_pct > 50 else (x1 > current_exit)
                            else:
                                near_check = (y1 < current_exit) if effective_pct > 50 else (y1 > current_exit)
                            if near_check:
                                pkg["pre_line_seen"] = True

                        # ── Crossing: leading edge past the line AND was on near side ──
                        # Inverted → travel is high→low; leading edge is x1/y1 (min coord).
                        # Normal  → travel is low→high; leading edge is x2/y2 (max coord).
                        if self._exit_line_inverted:
                            crossed_check = (x1 <= current_exit) if line_is_vert else (y1 <= current_exit)
                        else:
                            crossed_check = (x2 >= current_exit) if line_is_vert else (y2 >= current_exit)
                        if crossed_check and pkg["pre_line_seen"] and tid not in self.packets_crossed_line:
                            if pkg["decision_locked"]:
                                self.packets_crossed_line.add(tid)
                                continue
                            self.packets_crossed_line.add(tid)
                            self.total_packets += 1
                            self.packet_numbers[tid] = self.total_packets
                            has_bc = pkg["barcode_detected"]
                            has_dt = pkg.get("date_detected", False)
                            if self._use_secondary_date:
                                # Dual validation
                                final = "OK" if (has_bc and has_dt) else "NOK"
                                reasons = []
                                if has_bc: reasons.append("BARCODE")
                                else: reasons.append("NO BARCODE")
                                if has_dt: reasons.append("DATE")
                                else: reasons.append("NO DATE")
                                reason = " + ".join(reasons)
                            else:
                                final = "OK" if has_bc else "NOK"
                                reason = "BARCODE" if has_bc else "NO BARCODE"
                            pkg["decision_locked"] = True
                            pkg["final_decision"] = final
                            self.output_fifo.append(final)

                            # ── Save image locally for NOK packets only ──
                            if final == "NOK":
                                try:
                                    # Determine defect type
                                    if self._use_secondary_date:
                                        if not has_bc and not has_dt:
                                            defect_type = "both"
                                        elif not has_bc:
                                            defect_type = "no_barcode"
                                        elif not has_dt:
                                            defect_type = "no_date"
                                        else:
                                            defect_type = "unknown"
                                    else:
                                        defect_type = "no_barcode" if not has_bc else "unknown"

                                    # Create folder structure: images/{session_id}/{defect_type}/
                                    session_dir = os.path.join(os.path.dirname(__file__), "images", str(self._db_session_id))
                                    defect_dir = os.path.join(session_dir, defect_type)
                                    os.makedirs(defect_dir, exist_ok=True)

                                    # Save image as {packet_number}.jpg
                                    image_filename = f"{self.total_packets}.jpg"
                                    image_path = os.path.join(defect_dir, image_filename)
                                    cv2.imwrite(image_path, frame)

                                    # Record image in database
                                    relative_path = f"{self._db_session_id}/{defect_type}/{image_filename}"
                                    if self._db_writer:
                                        self._db_writer.record_image(self._db_session_id, self.total_packets, defect_type, relative_path)

                                    print(f"[IMAGE] Saved NOK packet {self.total_packets} to {defect_type} folder")
                                except Exception as img_err:
                                    print(f"[IMAGE] Error saving image: {img_err}")

                            print(f"[DET] Packet #{self.total_packets} -> {final} ({reason})")

                            # ── Compute defect type for DB ──
                            if self._use_secondary_date:
                                if not has_bc and not has_dt:
                                    _defect = "both"
                                elif not has_bc:
                                    _defect = "no_barcode"
                                elif not has_dt:
                                    _defect = "no_date"
                                else:
                                    _defect = None
                            else:
                                _defect = "no_barcode" if not has_bc else None

                            # ── Update in-memory NOK sub-counters ──
                            if _defect == "no_barcode":
                                self._nok_no_barcode += 1
                            elif _defect == "no_date":
                                self._nok_no_date += 1
                            elif _defect == "both":
                                self._nok_both += 1

                            # ── Periodic KPI snapshot ──
                            if (self._db_writer and self._stats_active
                                    and self._db_session_id
                                    and self.total_packets % SNAPSHOT_EVERY_N_PACKETS == 0):
                                try:
                                    _ok = self.output_fifo.count("OK")
                                    self._db_writer.write_queue.put_nowait({
                                        "type":           "snapshot",
                                        "session_id":     self._db_session_id,
                                        "total":          self.total_packets,
                                        "ok_count":       _ok,
                                        "nok_no_barcode": self._nok_no_barcode,
                                        "nok_no_date":    self._nok_no_date,
                                        "nok_both":       self._nok_both,
                                    })
                                except Exception:
                                    pass

                # ── Detection timing ──
                det_ms = (time.time() - t_start) * 1000
                det_fps = 1000 / det_ms if det_ms > 0 else 0

                # ── Build FIFO string ──
                fifo_items = []
                for i, d in enumerate(self.output_fifo[-8:],
                                      start=max(1, self.total_packets - 7)):
                    fifo_items.append(f"#{i}:{d}")
                fifo_str = " | ".join(fifo_items) if fifo_items else "(empty)"

                barcode_vis = [(int(bx1), int(by1), int(bx2), int(by2), bc)
                               for bx1, by1, bx2, by2, bc in barcode_dets]

                date_vis = [(dx1, dy1, dx2, dy2, dc)
                            for dx1, dy1, dx2, dy2, dc in all_date_dets]

                with self._overlay_lock:
                    self._overlay = {
                        'track_boxes': track_boxes,
                        'barcode_boxes': barcode_vis,
                        'date_boxes': date_vis,
                        'exit_line_y': self._exit_line_y,
                        'total_packets': self.total_packets,
                        'fifo_str': fifo_str,
                        'det_fps': det_fps,
                        'det_ms': det_ms,
                        'frame_idx': frame_idx,
                    }
                    self._overlay_frame = frame

                ok = self.output_fifo.count("OK")
                nok = self.output_fifo.count("NOK")

                with self._stats_lock:
                    self.stats.update({
                        "det_fps": round(det_fps, 1),
                        "inference_ms": round(det_ms, 1),
                        "total_packets": self.total_packets,
                        "packages_ok": ok,
                        "packages_nok": nok,
                        "rotation_deg": (self._rotation_steps % 4) * 90,
                        "fifo_queue": list(self.output_fifo[-10:]),
                    })

            print("[DETECTOR] Stopped")

        except Exception as e:
            print(f"[DETECTOR] Error: {e}")
            import traceback
            traceback.print_exc()

    # ═══════════════════════════════════════════
    # THREAD 3: COMPOSITOR (pre-encode JPEG)
    # ═══════════════════════════════════════════

    def _compositor_loop(self):
        """
        Continuously composites raw frame + detection overlay and
        pre-encodes to JPEG bytes. The MJPEG feed just yields these
        bytes instantly — zero computation in the request handler.
        """
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        print("[COMPOSITOR] Started")

        try:
            while self.is_running or self._video_ended:
                got = self._raw_changed.wait(timeout=0.1)
                if not self.is_running and not self._video_ended:
                    break
                if got:
                    self._raw_changed.clear()

                # For video files use the frame that was actually detected on
                # so bounding boxes align exactly with the pixels they describe.
                # For live cameras keep using the freshest raw frame.
                if self._is_video_file:
                    with self._overlay_lock:
                        det_frame = self._overlay_frame
                    if det_frame is not None:
                        frame = det_frame.copy()
                    else:
                        with self._raw_lock:
                            raw = self._raw_frame
                        if raw is None:
                            continue
                        frame = raw.copy()
                else:
                    with self._raw_lock:
                        raw = self._raw_frame
                    if raw is None:
                        continue
                    frame = raw.copy()
                h, w = frame.shape[:2]

                # ── RAW MODE: skip all detection overlays ──
                if self._raw_mode:
                    cv2.putText(frame, "RAW", (w - 100, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
                    cv2.putText(frame, f"Frame: {self.frame_count}",
                                (w - 180, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                    if self._paused:
                        cv2.putText(frame, "|| PAUSED", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    ret, buf = cv2.imencode('.jpg', frame, encode_params)
                    if ret:
                        with self._jpeg_lock:
                            self._jpeg_bytes = buf.tobytes()
                    if self._video_ended and not self.is_running:
                        break
                    continue

                # Grab latest overlay
                with self._overlay_lock:
                    ov_tracks    = list(self._overlay.get('track_boxes', []))
                    ov_barcodes  = list(self._overlay.get('barcode_boxes', []))
                    ov_dates     = list(self._overlay.get('date_boxes', []))
                    ov_total     = self._overlay.get('total_packets', 0)
                    ov_fifo      = self._overlay.get('fifo_str', '(empty)')
                    ov_det_fps   = self._overlay.get('det_fps', 0)
                    ov_det_ms    = self._overlay.get('det_ms', 0)
                    ov_ad_zones  = self._overlay.get('ad_zone_lines', None)

                ov_ely = self._exit_line_y
                ov_line_vert = self._exit_line_vertical
                if ov_ely <= 0:
                    ref = w if ov_line_vert else h
                    if ref > 0:
                        ov_ely = int(ref * self._exit_line_pct / 100)

                # ── Draw detection boxes ──
                for (x1, y1, x2, y2, label, color) in ov_tracks:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # ── Draw barcode boxes ──
                for (bx1, by1, bx2, by2, bc) in ov_barcodes:
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                    cv2.putText(frame, f"barcode {bc:.2f}",
                                (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # ── Draw date boxes (black) ──
                for (dx1, dy1, dx2, dy2, dc) in ov_dates:
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 0), 2)
                    cv2.putText(frame, f"date {dc:.2f}",
                                (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # ── Draw exit line (horizontal or vertical) ──
                if ov_ely > 0 and self._exit_line_enabled:
                    if ov_line_vert:
                        cv2.line(frame, (ov_ely, 0), (ov_ely, h), (255, 0, 0), 3)
                        cv2.putText(frame, "EXIT", (ov_ely + 6, 36),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        cv2.line(frame, (0, ov_ely), (w, ov_ely), (255, 0, 0), 3)
                        cv2.putText(frame, "EXIT LINE", (w - 200, ov_ely - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # ── Draw anomaly detection zone lines (vertical) ──
                if ov_ad_zones is not None:
                    zs, ze = ov_ad_zones
                    cv2.line(frame, (zs, 0), (zs, h), (255, 0, 255), 2)
                    cv2.line(frame, (ze, 0), (ze, h), (255, 0, 255), 2)
                    cv2.putText(frame, "SCAN START", (ze + 6, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    cv2.putText(frame, "SCAN END", (zs + 6, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # ── HUD ──
                cv2.putText(frame, f"FIFO: {ov_fifo}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"TOTAL: {ov_total}",
                            (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if ov_det_ms == 0 and len(ov_tracks) == 0:
                    cv2.putText(frame, "YOLO: warming up...",
                                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                else:
                    cv2.putText(frame,
                                f"YOLO: {ov_det_ms:.0f}ms | ~{ov_det_fps:.0f}fps",
                                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.putText(frame, f"Frame: {self.frame_count}",
                            (w - 180, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                # Pause indicator
                if self._paused:
                    cv2.putText(frame, "|| PAUSED", (w // 2 - 80, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

                # ── Encode JPEG ──
                ret, buf = cv2.imencode('.jpg', frame, encode_params)
                if ret:
                    with self._jpeg_lock:
                        self._jpeg_bytes = buf.tobytes()

                if self._video_ended and not self.is_running:
                    break

        except Exception as e:
            print(f"[COMPOSITOR] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[COMPOSITOR] Stopped")
