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

from helpers import calculate_bbox_metrics
from tracking_config import (
    CONFIG, VIDEO_EXTENSIONS, JPEG_QUALITY,
    CAMERA_FPS, CAMERA_WIDTH, CAMERA_HEIGHT,
    DETECTOR_FRAME_SKIP,
    TRACKER_CONFIG,
)


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

        # Active checkpoint info (set by switch_checkpoint / init_models)
        self.mode = "tracking"          # "tracking" or "date"
        self.current_checkpoint = None  # the checkpoint dict from CHECKPOINTS

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
        """
        steps = self._rotation_steps % 4
        transposed = steps in (1, 3)
        if self._exit_line_vertical:
            ref = self._frame_height if transposed else self._frame_width
        else:
            ref = self._frame_width if transposed else self._frame_height
        if ref > 0:
            self._exit_line_y = int(ref * self._exit_line_pct / 100)

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

    def stop_processing(self):
        self.is_running = False
        self._paused = False
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
        if not self.is_running:
            return {"error": "Not running"}
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
            self.stop_processing()

        # 2. Unload model from VRAM
        if self.model is not None:
            print(f"[SWITCH] Unloading model from VRAM...")
            del self.model
            self.model = None
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
        # attempt FP16 on CUDA
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

        print(f"[SWITCH] Loaded | mode={self.mode} | "
              f"package_id={self.package_id} barcode_id={self.barcode_id} date_id={self.date_id}")

        # 4b. Apply default rotation for this checkpoint
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
                    if self.mode == "tracking":
                        results = self.model.track(
                            frame,
                            conf=min(CONFIG["conf_paquet"], CONFIG["conf_barcode"]),
                            imgsz=CONFIG["imgsz"],
                            verbose=False,
                            persist=True,
                            tracker=TRACKER_YAML_PATH,
                        )[0]
                    else:
                        results = self.model(
                            frame,
                            conf=min(CONFIG["conf_paquet"], CONFIG["conf_barcode"]),
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
                    with self._stats_lock:
                        self.stats.update({
                            "det_fps":       round(det_fps, 1),
                            "inference_ms":  round(det_ms, 1),
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
                        if self.package_id is not None and cls == self.package_id and conf >= CONFIG["conf_paquet"]:
                            tid = int(box_ids[i]) if box_ids is not None else -1
                            if tid >= 0:
                                tracks.append([int(x1), int(y1), int(x2), int(y2), tid])
                        elif self.barcode_id is not None and cls == self.barcode_id and conf >= CONFIG["conf_barcode"]:
                            barcode_dets.append([x1, y1, x2, y2, conf])
                        elif self.date_id is not None and cls == self.date_id and conf >= CONFIG.get("conf_date", 0.45):
                            date_dets.append([int(x1), int(y1), int(x2), int(y2), conf])

                # ── Per-track processing ──
                track_boxes = []

                for t in tracks:
                    x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])

                    if tid not in self.packages:
                        inherited_barcode = False
                        new_bbox = (x1, y1, x2, y2)
                        for etid, epkg in self.packages.items():
                            if epkg.get("barcode_detected") and epkg.get("prev_bbox"):
                                if self._compute_iou(new_bbox, epkg["prev_bbox"]) > 0.3:
                                    inherited_barcode = True
                                    print(f"[DET] Track {tid} inherited barcode from Track {etid} (IoU match)")
                                    break
                        self.packages[tid] = {
                            "barcode_detected": inherited_barcode,
                            "decision_locked": False,
                            "final_decision": None,
                            "prev_bbox": None,
                            "prev_area": None,
                            "frames_tracked": 0,
                            "first_frame": frame_idx,
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

                    # ── Visualization ──
                    if pkg["decision_locked"]:
                        color = (255, 165, 0)
                        status = pkg["final_decision"]
                    elif pkg["barcode_detected"]:
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
                    for t in tracks:
                        x1, y1, x2, y2, tid = map(int, t[:5])
                        if tid not in self.packages:
                            continue
                        pkg = self.packages[tid]
                        crossed_check = (x2 >= current_exit) if line_is_vert else (y2 >= current_exit)
                        if crossed_check and tid not in self.packets_crossed_line:
                            if pkg["decision_locked"]:
                                self.packets_crossed_line.add(tid)
                                continue
                            self.packets_crossed_line.add(tid)
                            self.total_packets += 1
                            self.packet_numbers[tid] = self.total_packets
                            final = "OK" if pkg["barcode_detected"] else "NOK"
                            pkg["decision_locked"] = True
                            pkg["final_decision"] = final
                            self.output_fifo.append(final)
                            reason = "BARCODE" if pkg["barcode_detected"] else "NO BARCODE"
                            print(f"[DET] Packet #{self.total_packets} -> {final} ({reason})")

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
                            for dx1, dy1, dx2, dy2, dc in date_dets]

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
