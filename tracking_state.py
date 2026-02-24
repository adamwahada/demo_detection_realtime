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

from helpers import calculate_bbox_metrics, detect_fallen_package
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
    Two independent threads:
      - Reader:   captures frames at native FPS (smooth video, no YOLO dependency)
      - Detector: runs YOLO + ByteTrack on latest frame, updates overlay & stats

    The video feed composites the latest raw frame with the latest detection
    overlay, so the stream is always smooth regardless of YOLO speed.
    """

    def __init__(self):
        self.model = None
        self.package_id = None
        self.barcode_id = None

        # Active checkpoint info (set by switch_checkpoint / init_models)
        self.mode = "tracking"          # "tracking" or "date"
        self.current_checkpoint = None  # the checkpoint dict from CHECKPOINTS

        # Session generation counter — prevents old reader threads from
        # clobbering is_running after a camera/checkpoint switch
        self._session_gen = 0

        self.video_source = None
        self.cap = None
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
        # ── Exit line enabled flag (can be toggled via API, survives sessions) ──
        self._exit_line_enabled = True

        # ── Per-session tracking state ──
        self.frame_count = 0
        self.packages = {}
        self.total_packets = 0
        self.output_fifo = []
        self.packet_numbers = {}
        self.packets_crossed_line = set()
        self.fallen_decided_tids = set()

        # ── Stats for API ──
        self._stats_lock = threading.Lock()
        self.stats = self._empty_stats()

    # ─────────────────────────────────────────

    @staticmethod
    def _empty_overlay():
        return {
            'track_boxes': [],
            'barcode_boxes': [],
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
            "packages_defective": 0,
            "fifo_queue": [],
            "is_running": False,
            "video_ended": False,
        }

    def _reset_session(self):
        self.packages = {}
        self.frame_count = 0
        self.total_packets = 0
        self.output_fifo = []
        self.packet_numbers = {}
        self.packets_crossed_line = set()
        self.fallen_decided_tids = set()
        self._video_ended = False
        self._raw_frame = None
        self._det_frame = None
        self._det_frame_idx = 0
        self._det_event.clear()
        self._raw_changed.clear()
        # Reset exit line so a different-resolution camera gets a fresh value;
        # the compositor fallback (from raw frame height) fills in immediately
        self._exit_line_y = 0
        with self._jpeg_lock:
            self._jpeg_bytes = None
        with self._overlay_lock:
            self._overlay = self._empty_overlay()
        with self._stats_lock:
            self.stats = self._empty_stats()

    # ═══════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════

    def start_processing(self, video_source):
        if self.is_running:
            return {"error": "Already processing"}

        self.video_source = video_source
        self._reset_session()
        self._is_video_file = any(
            video_source.lower().endswith(ext)
            for ext in VIDEO_EXTENSIONS
        )

        # Reset built-in tracker state for fresh session
        if hasattr(self.model, 'predictor') and self.model.predictor is not None:
            self.model.predictor.trackers = []
            self.model.predictor = None

        self._session_gen += 1          # bump before launching so threads capture the right id
        my_gen = self._session_gen
        self.is_running = True

        # Launch THREE parallel threads, stamping each with the current session gen
        threading.Thread(target=self._reader_loop,     args=(my_gen,), daemon=True, name="VideoReader").start()
        threading.Thread(target=self._detection_loop,  daemon=True, name="YOLODetector").start()
        threading.Thread(target=self._compositor_loop, daemon=True, name="Compositor").start()

        mode = "video_file" if self._is_video_file else "live"
        return {"status": "started", "source": video_source, "mode": mode}

    def stop_processing(self):
        self.is_running = False
        time.sleep(0.5)  # Give threads time to exit cleanly
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        # Force CUDA cleanup to prevent segfault
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        import gc
        gc.collect()
        with self._stats_lock:
            self.stats["is_running"] = False
        return {"status": "stopped"}

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
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            print(f"[SWITCH] VRAM freed.")

        # 3. Load new model
        print(f"[SWITCH] Loading checkpoint: {checkpoint['label']} ({checkpoint['path']})")
        self.model = YOLO(checkpoint["path"])
        self.model.to('cuda')
        names = self.model.names

        # 4. Resolve class IDs (None = class not present / not needed)
        pkg_cls = checkpoint.get("package_class")
        bar_cls = checkpoint.get("barcode_class")
        self.package_id = next((k for k, v in names.items() if v == pkg_cls), None) if pkg_cls else None
        self.barcode_id = next((k for k, v in names.items() if v == bar_cls), None) if bar_cls else None
        self.mode = checkpoint.get("mode", "tracking")
        self.current_checkpoint = checkpoint

        print(f"[SWITCH] Loaded | mode={self.mode} | "
              f"package_id={self.package_id} barcode_id={self.barcode_id}")

        # 5. Warm up
        try:
            import numpy as np
            from tracking_config import CONFIG
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, imgsz=CONFIG["imgsz"], verbose=False)
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
            if src.startswith("rtsp://"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif self._is_video_file:
                if not Path(src).exists():
                    print(f"[READER] ERROR: File not found: {src}")
                    self.is_running = False
                    return
                self.cap = cv2.VideoCapture(src)
            else:
                self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap or not self.cap.isOpened():
                print(f"[READER] ERROR: Cannot open source: {src}")
                self.is_running = False
                return

            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_time = 1.0 / fps

            with self._stats_lock:
                self.stats["video_fps"] = round(fps, 1)
                self.stats["is_running"] = True

            self._frame_width = w
            self._frame_height = h

            print(f"[READER] Opened: {w}x{h} @ {fps:.0f}fps | "
                  f"{'Video file' if self._is_video_file else 'Live camera'}")

            while self.is_running:
                t0 = time.time()

                ret, frame = self.cap.read()
                if not ret:
                    if self._is_video_file:
                        print("[READER] Video file ended")
                        self._video_ended = True
                        with self._stats_lock:
                            self.stats["video_ended"] = True
                    break

                self.frame_count += 1

                # Store raw frame for streaming (always latest)
                with self._raw_lock:
                    self._raw_frame = frame
                self._raw_changed.set()

                # ── Envoie 1 frame sur N au detector pour stabiliser l'inference ──
                if self.frame_count % DETECTOR_FRAME_SKIP == 0:
                    with self._det_lock:
                        self._det_frame = frame.copy()
                        self._det_frame_idx = self.frame_count
                    self._det_event.set()

                # Pour les fichiers vidéo: respecte le FPS natif
                if self._is_video_file:
                    elapsed = time.time() - t0
                    sleep = frame_time - elapsed
                    if sleep > 0:
                        time.sleep(sleep)

        except Exception as e:
            print(f"[READER] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            time.sleep(0.1)  # Let other threads notice is_running change
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            # Only update shared state if this is still the current session;
            # a newer session may have already set is_running = True.
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
            # Wait for first frame to get dimensions (up to 20s)
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
            EXIT_LINE_Y = int(height * (1 - CONFIG["exit_line_ratio"]))
            self._exit_line_y = EXIT_LINE_Y   # store on state — never through overlay

            # Immediately publish exit line so compositor can draw it
            with self._overlay_lock:
                self._overlay['exit_line_y'] = EXIT_LINE_Y

            print(f"[DETECTOR] Started | {width}x{height} | Exit Y={EXIT_LINE_Y}")

            last_processed_idx = 0

            while self.is_running:
                # Wait for new frame signal (longer timeout=ok, YOLO is slow)
                if not self._det_event.wait(timeout=1.0):
                    if not self.is_running:
                        break
                    continue
                self._det_event.clear()

                # Grab latest frame (always the newest, skip any we missed)
                with self._det_lock:
                    frame = self._det_frame
                    frame_idx = self._det_frame_idx

                if frame is None or frame_idx <= last_processed_idx:
                    continue

                last_processed_idx = frame_idx
                t_start = time.time()

                # ── YOLO Inference (with error handling) ──
                try:
                    if self.mode == "tracking":
                        # Built-in ByteTrack: detection + tracking in one call
                        results = self.model.track(
                            frame,
                            conf=min(CONFIG["conf_paquet"], CONFIG["conf_barcode"]),
                            imgsz=CONFIG["imgsz"],
                            verbose=False,
                            persist=True,
                            tracker=TRACKER_YAML_PATH,
                        )[0]
                    else:
                        # Date mode: plain detection, no tracking
                        results = self.model(
                            frame,
                            conf=min(CONFIG["conf_paquet"], CONFIG["conf_barcode"]),
                            imgsz=CONFIG["imgsz"],
                            verbose=False
                        )[0]
                except Exception as yolo_err:
                    print(f"[DETECTOR] YOLO inference error: {yolo_err}")
                    continue

                # ── DATE DETECTION mode: simple overlay, no tracking ──
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
                    last_processed_idx = frame_idx
                    continue   # skip tracking logic below

                # ── Extract tracked packages and barcode detections ──
                tracks = []
                barcode_dets = []

                if results.boxes is not None:
                    box_ids = results.boxes.id  # track IDs from built-in tracker (may be None)
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

                # ── Per-track processing ──
                track_boxes = []

                for t in tracks:
                    x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])

                    # Init new track
                    if tid not in self.packages:
                        self.packages[tid] = {
                            "barcode_detected": False,
                            "decision_locked": False,
                            "final_decision": None,
                            "is_fallen": False,
                            "prev_bbox": None,
                            "prev_area": None,
                            "healthy_area": None,
                            "frames_tracked": 0,
                            "first_frame": frame_idx,
                        }

                    pkg = self.packages[tid]

                    # ── Handle already-fallen debris ──
                    if pkg["is_fallen"] and pkg["decision_locked"]:
                        curr_area = (x2 - x1) * (y2 - y1)
                        prev_area = pkg["prev_area"] or 0
                        healthy_area = pkg["healthy_area"] or 0

                        area_jump = prev_area > 0 and (curr_area - prev_area) / prev_area > 0.40
                        center_jump = False
                        if pkg["prev_bbox"] is not None:
                            pcx = (pkg["prev_bbox"][0] + pkg["prev_bbox"][2]) / 2
                            pcy = (pkg["prev_bbox"][1] + pkg["prev_bbox"][3]) / 2
                            d = (((x1 + x2) / 2 - pcx) ** 2 + ((y1 + y2) / 2 - pcy) ** 2) ** 0.5
                            center_jump = d > max(width, height) * 0.12

                        is_new = ((healthy_area > 0 and curr_area > healthy_area * 0.75)
                                  or area_jump or center_jump)

                        if is_new:
                            print(f"[DET Frame {frame_idx}] Track {tid} REUSED: new package")
                            self.packets_crossed_line.discard(tid)
                            self.packet_numbers.pop(tid, None)
                            self.fallen_decided_tids.discard(tid)
                            self.packages[tid] = {
                                "barcode_detected": False,
                                "decision_locked": False,
                                "final_decision": None,
                                "is_fallen": False,
                                "prev_bbox": (x1, y1, x2, y2),
                                "prev_area": curr_area,
                                "healthy_area": None,
                                "frames_tracked": 1,
                                "first_frame": frame_idx,
                            }
                            pkg = self.packages[tid]
                        else:
                            # Still debris
                            pkg["prev_bbox"] = (x1, y1, x2, y2)
                            pkg["prev_area"] = curr_area
                            pn = self.packet_numbers.get(tid, '?')
                            track_boxes.append((x1, y1, x2, y2,
                                                f"#{pn} DEFECTIVE", (128, 128, 128)))
                            continue

                    pkg["frames_tracked"] += 1
                    bbox = (x1, y1, x2, y2)

                    # ── Fallen detection ──
                    is_fallen = detect_fallen_package(bbox, pkg["prev_bbox"])
                    if is_fallen and not pkg["is_fallen"]:
                        pkg["is_fallen"] = True
                        pkg["healthy_area"] = pkg["prev_area"]
                        if not pkg["decision_locked"]:
                            self.total_packets += 1
                            self.packet_numbers[tid] = self.total_packets
                            pkg["decision_locked"] = True
                            pkg["final_decision"] = "DEFECTIVE"
                            self.output_fifo.append("DEFECTIVE")
                            self.fallen_decided_tids.add(tid)
                            self.packets_crossed_line.add(tid)
                            print(f"[DET] Packet #{self.total_packets} -> DEFECTIVE (fallen)")

                        ca, _ = calculate_bbox_metrics(x1, y1, x2, y2)
                        pkg["prev_bbox"] = bbox
                        pkg["prev_area"] = ca
                        pn = self.packet_numbers.get(tid, '?')
                        track_boxes.append((x1, y1, x2, y2,
                                            f"#{pn} DEFECTIVE", (0, 0, 255)))
                        continue

                    # Update bbox metrics
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

                    # ── Visualization data ──
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

                # ── Exit line crossing ──
                if self._exit_line_enabled:
                    for t in tracks:
                        x1, y1, x2, y2, tid = map(int, t[:5])
                        if tid not in self.packages:
                            continue
                        pkg = self.packages[tid]
                        if y2 >= EXIT_LINE_Y and tid not in self.packets_crossed_line:
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

                # Barcode overlay
                barcode_vis = [(int(bx1), int(by1), int(bx2), int(by2), bc)
                               for bx1, by1, bx2, by2, bc in barcode_dets]

                # ── Store overlay for video feed ──
                with self._overlay_lock:
                    self._overlay = {
                        'track_boxes': track_boxes,
                        'barcode_boxes': barcode_vis,
                        'exit_line_y': EXIT_LINE_Y,
                        'total_packets': self.total_packets,
                        'fifo_str': fifo_str,
                        'det_fps': det_fps,
                        'det_ms': det_ms,
                        'frame_idx': frame_idx,
                    }

                # ── Update API stats ──
                ok = self.output_fifo.count("OK")
                nok = self.output_fifo.count("NOK")
                defective = self.output_fifo.count("DEFECTIVE")

                with self._stats_lock:
                    self.stats.update({
                        "det_fps": round(det_fps, 1),
                        "inference_ms": round(det_ms, 1),
                        "total_packets": self.total_packets,
                        "packages_ok": ok,
                        "packages_nok": nok,
                        "packages_defective": defective,
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
        Runs at native video FPS, woken by _raw_changed event.
        """
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        print("[COMPOSITOR] Started")

        try:
            while self.is_running or self._video_ended:
                # Block until reader produces a new frame (or timeout)
                got = self._raw_changed.wait(timeout=0.1)
                if not self.is_running and not self._video_ended:
                    break
                if got:
                    self._raw_changed.clear()

                # Grab latest raw frame
                with self._raw_lock:
                    raw = self._raw_frame
                if raw is None:
                    continue

                frame = raw.copy()
                h, w = frame.shape[:2]

                # Grab latest overlay (from detector thread)
                with self._overlay_lock:
                    ov_tracks    = list(self._overlay.get('track_boxes', []))
                    ov_barcodes  = list(self._overlay.get('barcode_boxes', []))
                    ov_total     = self._overlay.get('total_packets', 0)
                    ov_fifo      = self._overlay.get('fifo_str', '(empty)')
                    ov_det_fps   = self._overlay.get('det_fps', 0)
                    ov_det_ms    = self._overlay.get('det_ms', 0)

                # ── Exit line: use dedicated attribute (set once by detector, never overwritten) ──
                # Fall back to frame-based estimate only until detector computes it
                ov_ely = self._exit_line_y
                if ov_ely <= 0 and h > 0:
                    ov_ely = int(h * (1 - CONFIG["exit_line_ratio"]))

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

                # ── Draw exit line (always, even before first detection) ──
                if ov_ely > 0 and self._exit_line_enabled:
                    cv2.line(frame, (0, ov_ely), (w, ov_ely), (255, 0, 0), 3)
                    cv2.putText(frame, "EXIT LINE", (w - 200, ov_ely - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # ── HUD ──
                cv2.putText(frame, f"FIFO: {ov_fifo}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"TOTAL: {ov_total}",
                            (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Show "Waiting for YOLO..." before first detection arrives
                if ov_det_ms == 0 and len(ov_tracks) == 0:
                    cv2.putText(frame, "YOLO: warming up...",
                                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                else:
                    cv2.putText(frame,
                                f"YOLO: {ov_det_ms:.0f}ms | ~{ov_det_fps:.0f}fps",
                                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.putText(frame, f"Frame: {self.frame_count}",
                            (w - 180, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                # ── Encode to JPEG once (reused by all browser clients) ──
                ret, buf = cv2.imencode('.jpg', frame, encode_params)
                if ret:
                    with self._jpeg_lock:
                        self._jpeg_bytes = buf.tobytes()

                # If video ended, keep last frame encoded and exit
                if self._video_ended and not self.is_running:
                    break

        except Exception as e:
            print(f"[COMPOSITOR] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[COMPOSITOR] Stopped")
