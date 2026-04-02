"""
TrackingState — Parallel architecture for video reading, YOLO detection, and compositing.
Thread 1 (Reader):   Reads video at native FPS — smooth, never waits for YOLO
Thread 2 (Detector): Runs YOLO + ByteTrack in parallel — updates stats & overlays
Thread 3 (Compositor): Composites raw frame + detection overlay, encodes JPEG
"""

import os
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

try:
    from db_writer import DBWriter
    from db_config import SNAPSHOT_EVERY_N_PACKETS
    _DB_AVAILABLE = True
except Exception:
    DBWriter = None
    SNAPSHOT_EVERY_N_PACKETS = 25
    _DB_AVAILABLE = False

from helpers import calculate_bbox_metrics, letterbox_image
from tracking_config import (
    CONFIG, VIDEO_EXTENSIONS, JPEG_QUALITY,
    CAMERA_FPS, CAMERA_WIDTH, CAMERA_HEIGHT,
    DETECTOR_FRAME_SKIP,
    ANOMALY_FRAME_SKIP,
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

# ── Bounded thread pools (Fix 4 & Fix 6) ──
_secondary_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="SecModel")
_proof_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ProofSave")


class TrackingState:
    """
    Two independent threads:
      - Reader:   captures frames at native FPS (smooth video, no YOLO dependency)
      - Detector: runs YOLO + ByteTrack on latest frame, updates overlay & stats

    The video feed composites the latest raw frame with the latest detection
    overlay, so the stream is always smooth regardless of YOLO speed.
    """

    def __init__(self, pipeline_id=None, db_writer=None):
        self.pipeline_id = pipeline_id
        self.model = None
        self.package_id = None
        self.barcode_id = None
        self.date_id = None

        # Secondary date-detection model (loaded when checkpoint has
        # "secondary_date_model_path"). Runs in parallel for best accuracy.
        self.secondary_model = None
        self._secondary_date_id = None
        self._use_secondary_date = False

        # ── EfficientAD anomaly detection models (loaded for mode="anomaly") ──
        self._ad_teacher = None
        self._ad_student = None
        self._ad_autoencoder = None
        self._ad_mean = None
        self._ad_std = None
        self._ad_quantiles = None
        self._ad_transform = None
        self._ad_track_states = {}  # {track_id: {'results': [], 'decision': None}}

        # Active checkpoint info (set by switch_checkpoint / init_models)
        self.mode = "tracking"          # "tracking", "date", or "anomaly"
        self.current_checkpoint = None  # the checkpoint dict from CHECKPOINTS

        # Session generation counter — prevents old reader threads from
        # clobbering is_running after a camera/checkpoint switch
        self._session_gen = 0

        self.video_source = None
        self.cap = None
        self.is_running = False
        self._is_video_file = False
        self._video_ended = False

        # ── Pause/Resume state ──
        self._is_paused = False
        self._paused_frame_pos = 0   # saved cap frame pos for video file resume
        self._paused_source = None

        # ── Raw frame from reader (always latest, always smooth) ──
        self._raw_frame = None
        self._raw_lock = threading.Lock()
        self._raw_changed = threading.Event()
        # Keep a short history so compositor can draw overlays on the exact
        # frame used by detector, preventing visual box shift on fast motion.
        self._raw_history = deque(maxlen=24)
        self._raw_history_lock = threading.Lock()

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
        # ── Exit line as % from leading edge (survives sessions & rotation changes) ──
        self._exit_line_pct = 85
        # ── Exit line orientation: False = horizontal (y), True = vertical (x) ──
        self._exit_line_vertical = False
        # ── Exit line direction inverted: % measured from opposite edge ──
        # e.g. 85% normally = 85% from top; inverted = 85% from bottom (= 15% from top)
        self._exit_line_inverted = False
        # ── Frame rotation steps (0,1,2,3 => 0°,90°,180°,270° CCW; survives sessions) ──
        self._rotation_steps = 0

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
        self._perf_lock = threading.Lock()
        self._perf = self._empty_perf()

        # ── DB writer (fully async, never blocks detector/compositor) ──
        # When a db_writer is injected (multi-pipeline mode) use it;
        # otherwise create our own (single-pipeline backward compat).
        if db_writer is not None:
            self._db_writer = db_writer
        else:
            self._db_writer = DBWriter() if _DB_AVAILABLE else None
        self._db_writer_started = False
        self._db_session_id = None
        self._stats_active = False
        self._nok_no_barcode = 0
        self._nok_no_date = 0
        self._nok_anomaly = 0
        # Baselines captured when stats recording starts, so session
        # totals reflect only the recording window.
        self._session_baseline_total = 0
        self._session_baseline_ok = 0

        # ── Proof-image session (liveImages/<session>/<defect_type>/) ──
        # Proof saves are gated on _stats_active; folder = _db_session_id

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
            "is_paused": False,
            "video_ended": False,
        }

    @staticmethod
    def _empty_perf():
        return {
            "detector_lag_frames": 0,
            "detector_last_frame_idx": 0,
            "detector_loop_ms": 0.0,
            "compositor_loop_ms": 0.0,
            "compositor_sync_hits": 0,
            "compositor_sync_misses": 0,
            "raw_history_len": 0,
        }

    # ─────────────────────────────────────────
    # ANOMALY DETECTION HELPERS (lightweight / best-effort)
    # These implement a minimal, defensive loader so the demo can register
    # the EfficientAD artifacts present in the checkpoint config. Full
    # inference pipelines are left to the main app (tracking_live) but this
    # enables mode='anomaly' without crashes and attempts to load torch
    # artifacts if available.
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

    def _ad_detect_anomaly_batch(self, crops):
        """Run EfficientAD on a list of 256x256 RGB uint8 crops in one batched GPU call.
        Returns list of (is_defective, score) in same order as input crops."""
        import torch
        from PIL import Image
        from efficientad import predict as effpredict

        if not crops:
            return []

        device = next(self._ad_teacher.parameters()).device
        thresh = (self.current_checkpoint or {}).get("ad_thresh", 5000.0)

        # Build batch tensor on CPU then transfer once
        tensors = []
        sizes = []
        for crop_np in crops:
            pil_img = Image.fromarray(crop_np)
            sizes.append((pil_img.width, pil_img.height))
            tensors.append(self._ad_transform(pil_img))
        batch = torch.stack(tensors).to(device)  # [N, 3, 256, 256]

        # Single batched forward pass through all three networks
        map_combined, _, _ = effpredict(
            image=batch, teacher=self._ad_teacher, student=self._ad_student,
            autoencoder=self._ad_autoencoder, teacher_mean=self._ad_mean,
            teacher_std=self._ad_std,
            q_st_start=self._ad_quantiles['q_st_start'],
            q_st_end=self._ad_quantiles['q_st_end'],
            q_ae_start=self._ad_quantiles['q_ae_start'],
            q_ae_end=self._ad_quantiles['q_ae_end']
        )

        # Post-process per sample
        results = []
        for k in range(map_combined.shape[0]):
            m = map_combined[k:k+1]  # keep batch dim [1, 1, H, W]
            orig_w, orig_h = sizes[k]
            m = torch.nn.functional.pad(m, (4, 4, 4, 4))
            m = torch.nn.functional.interpolate(m, (orig_h, orig_w), mode='bilinear')
            score = m[0, 0].cpu().numpy().max()
            results.append((score > thresh, float(score)))

        return results

    @staticmethod
    def _ad_final_decision(results, strategy="MAJORITY"):
        """Aggregate per-frame anomaly results into a final decision."""
        if not results:
            return False
        if strategy == "OR":
            return any(results)
        # MAJORITY
        return sum(results) > (len(results) / 2)

    def _save_nok_packet(self, pkt_num, tstate, checkpoint, session_id=None):
        """Save NOK packet — worst-crop image + CSV under liveImages/<session>/anomalie/<pkt_num>/."""
        import csv
        if session_id:
            base = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "liveImages", session_id, "anomalie", str(pkt_num),
            )
        else:
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anomalie", str(pkt_num))
        pkt_dir = base
        try:
            os.makedirs(pkt_dir, exist_ok=True)
        except OSError as e:
            print(f"[AD] Cannot create {pkt_dir}: {e}")
            return

        thresh = checkpoint.get("ad_thresh", 5000.0)
        scores = tstate.get("scores", [])
        results = tstate.get("results", [])
        crops = tstate.get("crops", [])

        # Save worst crop image
        if crops and scores:
            worst_idx = max(range(len(scores)), key=lambda i: scores[i])
            img_path = os.path.join(pkt_dir, "worst_scan.png")
            try:
                bgr = cv2.cvtColor(crops[worst_idx], cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            except Exception as e:
                print(f"[AD] Failed to save {img_path}: {e}")

        # Write CSV summary
        csv_path = os.path.join(pkt_dir, "scans.csv")
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["scan", "score", "threshold", "is_defective"])
                for idx, (sc, defective) in enumerate(
                    zip(scores, results), start=1
                ):
                    writer.writerow([idx, f"{sc:.2f}", f"{thresh:.2f}",
                                     "YES" if defective else "NO"])
                writer.writerow([])
                writer.writerow(["DECISION", "NOK",
                                 f"defective_scans={sum(results)}/{len(results)}",
                                 f"strategy={checkpoint.get('ad_strategy', 'MAJORITY')}"])
            print(f"[AD] Saved NOK packet #{pkt_num} -> {pkt_dir}")
        except Exception as e:
            print(f"[AD] Failed to write CSV {csv_path}: {e}")

    def _save_nok_packet_bg(self, pkt_num, tstate, checkpoint):
        """Fire-and-forget: save NOK image+CSV via bounded thread pool."""
        # Snapshot data; capture session_id now so folder is stable even if session changes
        data = {
            'results': list(tstate.get('results', [])),
            'scores': list(tstate.get('scores', [])),
            'crops': list(tstate.get('crops', [])),
        }
        session_now = self._db_session_id if self._stats_active else None
        cp_copy = dict(checkpoint)
        _proof_executor.submit(
            self._save_nok_packet,
            pkt_num, data, cp_copy, session_now,
        )

    def _reset_session(self):
        self._is_paused = False
        self._paused_frame_pos = 0
        self._paused_source = None
        self.packages = {}
        self.frame_count = 0
        self.total_packets = 0
        self.output_fifo = []
        self.packet_numbers = {}
        self.packets_crossed_line = set()
        self._ad_track_states = {}
        self._video_ended = False
        self._raw_frame = None
        self._det_frame = None
        self._det_frame_idx = 0
        self._det_event.clear()
        self._raw_changed.clear()
        with self._raw_history_lock:
            self._raw_history.clear()
        # Reset exit line so a different-resolution camera gets a fresh value;
        # the compositor fallback (from raw frame height) fills in immediately
        self._exit_line_y = 0
        with self._jpeg_lock:
            self._jpeg_bytes = None
        with self._overlay_lock:
            self._overlay = self._empty_overlay()
        with self._stats_lock:
            self.stats = self._empty_stats()
            self.stats["rotation_deg"] = (self._rotation_steps % 4) * 90
        with self._perf_lock:
            self._perf = self._empty_perf()
        self._nok_no_barcode = 0
        self._nok_no_date = 0
        self._nok_anomaly = 0

    # ─────────────────────────────────────────
    # PROOF IMAGE SAVING (liveImages/<session>/<defect_type>/)
    # ─────────────────────────────────────────

    def _save_proof_image(self, pkt_num, defect_type, frame, bbox=None, session_id=None):
        """Save a proof image of a defective packet.

        Folder structure: liveImages/<session_id>/<defect_type>/packet_<N>.png
        Only called when a stats recording session is active.
        """
        if not session_id:
            return
        base = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "liveImages", session_id, defect_type,
        )
        try:
            os.makedirs(base, exist_ok=True)
        except OSError:
            return

        img = frame
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            # Add 15% padding around the bbox for context
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.15), int(bh * 0.15)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
            if x2 > x1 and y2 > y1:
                img = frame[y1:y2, x1:x2]

        img_path = os.path.join(base, f"packet_{pkt_num}.png")
        try:
            cv2.imwrite(img_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        except Exception as e:
            print(f"[PROOF] Failed to save {img_path}: {e}")

    def _save_proof_image_bg(self, pkt_num, defect_type, frame, bbox=None):
        """Fire-and-forget: save proof image via bounded thread pool.

        Only saves when a stats recording session is active.
        Captures session_id at scheduling time so the image always lands
        in the correct session folder even if recording is toggled.
        """
        if not self._stats_active or not self._db_session_id:
            return
        frame_copy = frame.copy()
        session_now = self._db_session_id
        _proof_executor.submit(
            self._save_proof_image,
            pkt_num, defect_type, frame_copy, bbox, session_now,
        )

    # ─────────────────────────────────────────

    def set_stats_recording(self, active, group_id="", shift_id=""):
        active = bool(active)
        if active == self._stats_active:
            return {"stats_active": self._stats_active, "session_id": self._db_session_id}

        if active:
            new_sid = None
            if self._db_writer:
                if not self._db_writer_started:
                    self._db_writer.start()
                    self._db_writer_started = True
                cp_id = (self.current_checkpoint or {}).get("id", "")
                cam_src = str(self.video_source or "")
                new_sid = self._db_writer.open_session(checkpoint_id=cp_id, camera_source=cam_src, group_id=group_id, shift_id=shift_id)
                self._db_writer.set_active(True)
            self._db_session_id = new_sid
            self._stats_active = True
            self._nok_no_barcode = 0
            self._nok_no_date = 0
            self._nok_anomaly = 0
            # Reset live counters so frontend display starts from 0 with the
            # recording session — packet numbers, images and DB rows all align.
            self.total_packets = 0
            self.output_fifo = []
            self.packages = {}
            self.packet_numbers = {}
            self.packets_crossed_line = set()
            self._session_baseline_total = 0
            self._session_baseline_ok = 0
            with self._stats_lock:
                self.stats["total_packets"] = 0
                self.stats["packages_ok"] = 0
                self.stats["packages_nok"] = 0
                self.stats["fifo_queue"] = []
            return {"stats_active": True, "session_id": new_sid}

        if self._db_writer and self._db_session_id:
            self._db_writer.close_session(self._db_session_id, totals=self._db_totals())
            self._db_writer.set_active(False)
        self._db_session_id = None
        self._stats_active = False
        return {"stats_active": False, "session_id": None}

    def _db_totals(self):
        return {
            "total": self.total_packets,
            "ok_count": self.output_fifo.count("OK"),
            "nok_no_barcode": self._nok_no_barcode,
            "nok_no_date": self._nok_no_date,
            "nok_anomaly": self._nok_anomaly,
        }

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

    @staticmethod
    def _intersection_over_box(box, container):
        """Return the fraction of `box` area that lies inside `container`."""
        ix1 = max(box[0], container[0])
        iy1 = max(box[1], container[1])
        ix2 = min(box[2], container[2])
        iy2 = min(box[3], container[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        return inter / area

    def _det_box_matches_package(self, det_box, pkg_box, kind):
        """Strict association for small inner detections like barcode/date.

        IoU alone is too weak here because barcode/date boxes are much smaller
        than the package box, so require most of the detection box to be inside
        the package as well.
        """
        cfg = CONFIG
        iou_min = cfg.get(f"{kind}_match_iou_min", 0.01)
        inside_min = cfg.get(f"{kind}_match_inside_min", 0.60)
        iou = self._compute_iou(det_box, pkg_box)
        inside = self._intersection_over_box(det_box, pkg_box)
        return iou >= iou_min and inside >= inside_min

    # ═══════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════

    def start_processing(self, video_source):
        if self.is_running:
            return {"error": "Already processing"}

        self.video_source = video_source
        self._reset_session()
        self._is_video_file = isinstance(video_source, str) and any(
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
        self._is_paused = False
        self._paused_frame_pos = 0
        self._paused_source = None
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        import gc
        gc.collect()
        with self._stats_lock:
            self.stats["is_running"] = False
            self.stats["is_paused"] = False
        return {"status": "stopped"}

    def pause_processing(self):
        """Pause: halt threads and save video position. Accumulated stats are preserved."""
        if not self.is_running:
            return {"status": "not_running", "is_paused": self._is_paused}

        # Save cap frame position before stopping (video files only)
        paused_pos = 0
        if self._is_video_file and self.cap:
            try:
                paused_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            except Exception:
                pass

        self.is_running = False
        time.sleep(0.5)

        if self.cap:
            try:
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

        self._is_paused = True
        self._paused_frame_pos = paused_pos
        self._paused_source = self.video_source

        with self._stats_lock:
            self.stats["is_running"] = False
            self.stats["is_paused"] = True

        print(f"[PAUSE] Paused at frame {paused_pos} | "
              f"total_packets={self.total_packets}")
        return {
            "status": "paused",
            "frame_pos": paused_pos,
            "total_packets": self.total_packets,
            "is_video_file": self._is_video_file,
        }

    def _reset_session_for_resume(self):
        """Partial reset for resume: preserve accumulated stats, clear transient state."""
        # Preserve these
        _total = self.total_packets
        _fifo = list(self.output_fifo)
        _pkt_numbers = dict(self.packet_numbers)
        _nok_bc = self._nok_no_barcode
        _nok_dt = self._nok_no_date
        _nok_ano = self._nok_anomaly

        # Reset transient tracking state
        self.packages = {}
        self.frame_count = 0
        self.packets_crossed_line = set()   # clear: new track IDs after restart
        self._ad_track_states = {}
        self._video_ended = False
        self._raw_frame = None
        self._det_frame = None
        self._det_frame_idx = 0
        self._det_event.clear()
        self._raw_changed.clear()
        with self._raw_history_lock:
            self._raw_history.clear()
        self._exit_line_y = 0
        with self._jpeg_lock:
            self._jpeg_bytes = None
        with self._overlay_lock:
            self._overlay = self._empty_overlay()

        # Restore preserved stats
        self.total_packets = _total
        self.output_fifo = _fifo
        self.packet_numbers = _pkt_numbers
        self._nok_no_barcode = _nok_bc
        self._nok_no_date = _nok_dt
        self._nok_anomaly = _nok_ano

        ok = _fifo.count("OK")
        nok = _fifo.count("NOK")
        with self._stats_lock:
            self.stats = self._empty_stats()
            self.stats["rotation_deg"] = (self._rotation_steps % 4) * 90
            self.stats["total_packets"] = _total
            self.stats["packages_ok"] = ok
            self.stats["packages_nok"] = nok
        with self._perf_lock:
            self._perf = self._empty_perf()

    def resume_processing(self):
        """Resume from a paused state without resetting accumulated stats."""
        if self.is_running:
            return {"error": "Already running"}
        if not self._is_paused:
            return {"status": "not_paused", "hint": "Use /api/start to start fresh"}

        source = self._paused_source
        if source is None:
            return {"error": "No paused source stored"}

        self.video_source = source
        self._is_paused = False

        # Partial reset: keep stats, clear tracking state
        self._reset_session_for_resume()

        # Reset YOLO tracker to ensure clean tracking
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
        print(f"[RESUME] Resuming from frame {self._paused_frame_pos} | "
              f"total_packets={self.total_packets}")
        return {
            "status": "resumed",
            "source": str(source),
            "mode": mode,
            "total_packets": self.total_packets,
        }

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

        # 2. Unload model(s) from VRAM
        if self.model is not None:
            print(f"[SWITCH] Unloading model from VRAM...")
            del self.model
            self.model = None
        if self.secondary_model is not None:
            print(f"[SWITCH] Unloading secondary date model from VRAM...")
            del self.secondary_model
            self.secondary_model = None
            self._secondary_date_id = None
            self._use_secondary_date = False
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
        self.model.to(DEVICE)
        names = self.model.names

        # 4. Resolve class IDs (None = class not present / not needed)
        pkg_cls = checkpoint.get("package_class")
        bar_cls = checkpoint.get("barcode_class")
        date_cls = checkpoint.get("date_class")
        self.package_id = next((k for k, v in names.items() if v == pkg_cls), None) if pkg_cls else None
        self.barcode_id = next((k for k, v in names.items() if v == bar_cls), None) if bar_cls else None
        self.date_id = next((k for k, v in names.items() if v == date_cls), None) if date_cls else None
        self.mode = checkpoint.get("mode", "tracking")
        self.current_checkpoint = checkpoint

        # Apply per-checkpoint exit line settings if present
        if "exit_line_pct" in checkpoint:
            self._exit_line_pct = checkpoint["exit_line_pct"]
        if "exit_line_vertical" in checkpoint:
            self._exit_line_vertical = checkpoint["exit_line_vertical"]
        if "exit_line_inverted" in checkpoint:
            self._exit_line_inverted = checkpoint["exit_line_inverted"]
        if self._frame_height > 0 or self._frame_width > 0:
            self._recompute_exit_line_y()

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
            sec_names = self.secondary_model.names
            self._secondary_date_id = next(
                (k for k, v in sec_names.items() if v == sec_cls), None
            ) if sec_cls else None
            self._use_secondary_date = self._secondary_date_id is not None
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

        # 5. Warm up
        try:
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
            if isinstance(src, str) and src.startswith("rtsp://"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif self._is_video_file:
                try:
                    file_exists = Path(src).exists()
                except (PermissionError, OSError):
                    file_exists = False
                if not file_exists:
                    print(f"[READER] ERROR: File not found or not accessible: {src}")
                    self.is_running = False
                    return
                self.cap = cv2.VideoCapture(src)
            else:
                import platform
                if isinstance(src, str) and src.isdigit():
                    src = int(src)
                if platform.system() == "Windows":
                    self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
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

            raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
            # Live cameras often report 0; use requested CAMERA_FPS as fallback
            fps = raw_fps if raw_fps and raw_fps > 0 else CAMERA_FPS
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_time = 1.0 / fps

            # Seek to saved position when resuming a paused video file
            if self._is_video_file and self._paused_frame_pos > 0:
                print(f"[READER] Resuming from frame {self._paused_frame_pos}")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._paused_frame_pos)
                self._paused_frame_pos = 0

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
                        # Loop the video for simulation purposes
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        if not ret:
                            print("[READER] Video file ended (could not loop)")
                            self._video_ended = True
                            with self._stats_lock:
                                self.stats["video_ended"] = True
                            break
                        print("[READER] Video looped back to start")
                    else:
                        break

                self.frame_count += 1
                frame_idx = self.frame_count

                # Optional live rotation (applies to stream + detector)
                rot_steps = self._rotation_steps % 4
                if rot_steps:
                    frame = self._rotate_frame_ccw(frame, rot_steps)

                # Store raw frame for streaming (always latest)
                with self._raw_lock:
                    self._raw_frame = frame
                with self._raw_history_lock:
                    self._raw_history.append((frame_idx, frame))
                    history_len = len(self._raw_history)
                with self._perf_lock:
                    self._perf["raw_history_len"] = history_len
                self._raw_changed.set()

                # ── Envoie 1 frame sur N au detector pour stabiliser l'inference ──
                # ── Send 1 frame out of N to detector (mode-aware skip) ──
                _effective_skip = ANOMALY_FRAME_SKIP if self.mode == "anomaly" else DETECTOR_FRAME_SKIP
                if self.frame_count % _effective_skip == 0:
                    with self._det_lock:
                        self._det_frame = frame.copy()
                        self._det_frame_idx = frame_idx
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
            # Per-checkpoint exit line settings override the global CONFIG default
            cp = self.current_checkpoint or {}
            if "exit_line_pct" in cp:
                self._exit_line_pct = cp["exit_line_pct"]
            else:
                self._exit_line_pct = round((1.0 - CONFIG["exit_line_ratio"]) * 100)
            if "exit_line_vertical" in cp:
                self._exit_line_vertical = cp["exit_line_vertical"]
            if "exit_line_inverted" in cp:
                self._exit_line_inverted = cp["exit_line_inverted"]
            self._recompute_exit_line_y()
            EXIT_LINE_Y = self._exit_line_y

            # Immediately publish exit line so compositor can draw it
            with self._overlay_lock:
                self._overlay['exit_line_y'] = EXIT_LINE_Y

            steps = self._rotation_steps % 4
            orientation = 'horizontal' if steps in (0, 2) else 'vertical'
            print(f"[DETECTOR] Started | {width}x{height} | Exit={EXIT_LINE_Y}px "
                  f"({self._exit_line_pct}% | {orientation})")

            last_processed_idx = 0

            # ── YOLO + tracker warmup on the real first frame ──
            # This ensures ByteTrack internal state is initialised and GPU
            # kernels are compiled BEFORE we process real packets, avoiding
            # the first-packet lag that caused scans=0 / scans=1.
            try:
                cp = self.current_checkpoint or {}
                yolo_imgsz = cp.get("yolo_imgsz", CONFIG["imgsz"])
                yolo_conf = cp.get("yolo_conf", min(
                    CONFIG.get("conf_paquet", 0.45),
                    CONFIG.get("conf_barcode", 0.45),
                ))
                if self.mode in ("tracking", "anomaly"):
                    _warmup_kw = dict(
                        half=False, conf=yolo_conf, imgsz=yolo_imgsz,
                        verbose=False, persist=True, tracker=TRACKER_YAML_PATH,
                    )
                    if self.mode == "anomaly":
                        _warmup_kw["retina_masks"] = True
                    self.model.track(first_frame, **_warmup_kw)
                    print("[DETECTOR] Tracker warmup done (first .track() call)")
                else:
                    self.model(first_frame, conf=yolo_conf, imgsz=yolo_imgsz, verbose=False)
                    print("[DETECTOR] YOLO warmup done")
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as wu_err:
                print(f"[DETECTOR] Warmup failed (non-fatal): {wu_err}")

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
                lag_frames = max(0, self.frame_count - frame_idx)
                with self._perf_lock:
                    self._perf["detector_lag_frames"] = lag_frames
                    self._perf["detector_last_frame_idx"] = frame_idx

                # ── YOLO Inference (with error handling) ──
                try:
                    cp = self.current_checkpoint or {}
                    conf_list = [
                        v for k in ("conf_paquet", "conf_barcode", "conf_date")
                        if (v := cp.get(k)) is not None
                    ]
                    conf_min = min(conf_list) if conf_list else min(
                        CONFIG.get("conf_paquet", 0.45),
                        CONFIG.get("conf_barcode", 0.45),
                    )
                    yolo_imgsz = cp.get("yolo_imgsz", CONFIG["imgsz"])
                    yolo_conf = cp.get("yolo_conf", conf_min)

                    if self.mode == "tracking":
                        # Built-in ByteTrack: detection + tracking in one call
                        results = self.model.track(
                            frame,
                            half=False,
                            conf=conf_min,
                            imgsz=yolo_imgsz,
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
                        # Date mode: plain detection, no tracking
                        results = self.model(
                            frame,
                            conf=conf_min,
                            imgsz=yolo_imgsz,
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
                    with self._perf_lock:
                        self._perf["detector_loop_ms"] = round(det_ms, 2)
                    last_processed_idx = frame_idx
                    continue   # skip tracking logic below

                # ── ANOMALY DETECTION mode ──
                if self.mode == "anomaly":
                    cp = self.current_checkpoint or {}
                    zone_start_pct = cp.get("zone_start_pct", 0.20)
                    zone_end_pct = cp.get("zone_end_pct", 0.60)
                    ad_strategy = cp.get("ad_strategy", "MAJORITY")
                    ad_max_scans = cp.get("ad_max_scans", 5)
                    h_f, w_f = frame.shape[:2]
                    # Packets flow RIGHT → LEFT:
                    #   zone_end_px   = ENTRY line (right, where scanning begins)
                    #   zone_start_px = EXIT  line (left, decision is final)
                    exit_line_px = int(w_f * zone_start_pct)
                    entry_line_px = int(w_f * zone_end_pct)

                    track_boxes = []
                    ad_zone_lines = (exit_line_px, entry_line_px)

                    has_masks = results.masks is not None
                    has_ids = results.boxes is not None and results.boxes.id is not None

                    if has_masks and has_ids:
                        masks = results.masks.data.cpu().numpy()
                        boxes = results.boxes.xyxy.cpu().numpy()
                        track_ids = results.boxes.id.int().cpu().tolist()

                        # ── PHASE 1: Classify each track + collect crops (CPU only) ──
                        ad_batch_crops = []    # crops to send to EfficientAD
                        ad_batch_indices = []  # index into track_ids for each crop
                        per_track_info = []    # (i, tid, x1, y1, x2, y2, zone) per track

                        for i, tid in enumerate(track_ids):
                            if tid not in self._ad_track_states:
                                self._ad_track_states[tid] = {
                                    'results': [], 'scores': [],
                                    'crops': [], 'decision': None,
                                }
                            tstate = self._ad_track_states[tid]

                            x1, y1, x2, y2 = map(int, boxes[i])
                            center_x = (x1 + x2) // 2

                            if tstate['decision'] is not None:
                                per_track_info.append((i, tid, x1, y1, x2, y2, 'decided'))
                            elif center_x > entry_line_px:
                                per_track_info.append((i, tid, x1, y1, x2, y2, 'entering'))
                            elif exit_line_px <= center_x <= entry_line_px:
                                n_scans = len(tstate['results'])
                                if n_scans < ad_max_scans:
                                    img_crop = self._ad_crop_and_mask(frame, masks[i], cp)
                                    if img_crop is not None:
                                        ad_batch_crops.append(img_crop)
                                        ad_batch_indices.append(i)
                                per_track_info.append((i, tid, x1, y1, x2, y2, 'scanning'))
                            else:
                                per_track_info.append((i, tid, x1, y1, x2, y2, 'exiting'))

                        # ── PHASE 2: Batched EfficientAD inference (single GPU call) ──
                        ad_batch_results = []
                        if ad_batch_crops:
                            try:
                                ad_batch_results = self._ad_detect_anomaly_batch(ad_batch_crops)
                            except Exception as ad_err:
                                print(f"[AD] Batch inference error: {ad_err}")
                                ad_batch_results = [None] * len(ad_batch_crops)

                        # Build lookup: track index → (is_def, score)
                        ad_result_by_idx = {}
                        for batch_pos, track_idx in enumerate(ad_batch_indices):
                            ad_result_by_idx[track_idx] = ad_batch_results[batch_pos] if batch_pos < len(ad_batch_results) else None

                        # ── PHASE 3: Assign results and build overlay ──
                        for (i, tid, x1, y1, x2, y2, zone) in per_track_info:
                            tstate = self._ad_track_states[tid]
                            label = "WAITING"
                            color = (200, 200, 200)

                            if zone == 'decided':
                                is_def = tstate['decision']
                                pkt_num = self.packet_numbers.get(tid)
                                if is_def:
                                    label = f"#{pkt_num} DEFECTIVE" if pkt_num else f"T{tid} DEFECTIVE"
                                    color = (0, 0, 255)
                                else:
                                    label = f"#{pkt_num} GOOD" if pkt_num else f"T{tid} GOOD"
                                    color = (0, 255, 0)

                            elif zone == 'entering':
                                label = f"T{tid} ENTERING"
                                color = (200, 200, 200)

                            elif zone == 'scanning':
                                # Apply batched result if this track had a crop
                                batch_result = ad_result_by_idx.get(i)
                                if batch_result is not None:
                                    is_def, score = batch_result
                                    tstate['results'].append(is_def)
                                    tstate['scores'].append(score)
                                    tstate['crops'].append(ad_batch_crops[ad_batch_indices.index(i)].copy())
                                elif batch_result is None and i in ad_result_by_idx:
                                    # Batch error for this crop
                                    label = f"T{tid} AD-ERR"
                                    color = (0, 165, 255)
                                    track_boxes.append((x1, y1, x2, y2, label, color))
                                    continue
                                n_scans = len(tstate['results'])
                                label = f"T{tid} SCAN {n_scans}/{ad_max_scans}"
                                color = (0, 255, 255)

                            else:  # exiting
                                tstate['decision'] = self._ad_final_decision(
                                    tstate['results'], strategy=ad_strategy)
                                is_def = tstate['decision']

                                self.packets_crossed_line.add(tid)
                                self.total_packets += 1
                                self.packet_numbers[tid] = self.total_packets
                                final = "NOK" if is_def else "OK"
                                self.output_fifo.append(final)

                                if self._stats_active and is_def:
                                    self._nok_anomaly += 1

                                if is_def:
                                    print(f"[AD] Packet #{self.total_packets} -> NOK "
                                          f"(scans={len(tstate['results'])})")

                                if is_def:
                                    self._save_nok_packet_bg(
                                        self.total_packets, tstate, cp)

                                tstate['crops'] = []
                                tstate['scores'] = []

                                if self._stats_active:
                                    if (
                                        self._db_writer
                                        and self._db_session_id
                                        and self.total_packets % SNAPSHOT_EVERY_N_PACKETS == 0
                                    ):
                                        try:
                                            self._db_writer.write_queue.put_nowait({
                                                "type": "session_update",
                                                "session_id": self._db_session_id,
                                                "total": self.total_packets,
                                                "ok_count": self.output_fifo.count("OK"),
                                                "nok_no_barcode": 0,
                                                "nok_no_date": 0,
                                                "nok_anomaly": self._nok_anomaly,
                                            })
                                        except Exception:
                                            pass
                                    # Record defective packet with timestamp for ejection
                                    if is_def:
                                        try:
                                            from datetime import datetime
                                            self._db_writer.write_queue.put_nowait({
                                                "type": "crossing",
                                                "session_id": self._db_session_id,
                                                "packet_num": self.total_packets - self._session_baseline_total,
                                                "defect_type": "anomaly",
                                                "crossed_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
                                            })
                                        except Exception:
                                            pass

                                pkt_num = self.packet_numbers.get(tid)
                                if is_def:
                                    label = f"#{pkt_num} DEFECTIVE" if pkt_num else f"T{tid} DEFECTIVE"
                                    color = (0, 0, 255)
                                else:
                                    label = f"#{pkt_num} GOOD" if pkt_num else f"T{tid} GOOD"
                                    color = (0, 255, 0)

                            track_boxes.append((x1, y1, x2, y2, label, color))

                    det_ms = (time.time() - t_start) * 1000
                    det_fps = 1000 / det_ms if det_ms > 0 else 0

                    ok = self.output_fifo.count("OK")
                    nok = self.output_fifo.count("NOK")

                    # Build FIFO string from actual results (same as tracking mode)
                    fifo_items = []
                    for fi, fd in enumerate(self.output_fifo[-8:],
                                            start=max(1, self.total_packets - 7)):
                        fifo_items.append(f"#{fi}:{fd}")
                    fifo_str = " | ".join(fifo_items) if fifo_items else "(anomaly mode)"

                    with self._overlay_lock:
                        self._overlay = {
                            'track_boxes': track_boxes,
                            'barcode_boxes': [],
                            'exit_line_y': self._exit_line_y,
                            'total_packets': self.total_packets,
                            'fifo_str': fifo_str,
                            'det_fps': det_fps,
                            'det_ms': det_ms,
                            'frame_idx': frame_idx,
                            'ad_zone_lines': ad_zone_lines,
                        }
                    with self._stats_lock:
                        self.stats.update({
                            "det_fps": round(det_fps, 1),
                            "inference_ms": round(det_ms, 1),
                            "total_packets": self.total_packets,
                            "packages_ok": ok,
                            "packages_nok": nok,
                            "fifo_queue": list(self.output_fifo[-10:]),
                        })
                    with self._perf_lock:
                        self._perf["detector_loop_ms"] = round(det_ms, 2)
                    continue

                # ── Extract tracked packages, barcode and date detections ──
                tracks = []
                barcode_dets = []
                date_dets = []

                # ── Submit secondary date model in parallel (Fix 4) ──
                sec_future = None
                if self._use_secondary_date and self.secondary_model is not None:
                    sec_conf = self.current_checkpoint.get("conf_date", CONFIG.get("conf_date", 0.30))
                    sec_future = _secondary_executor.submit(
                        self.secondary_model,
                        frame,
                        conf=sec_conf,
                        imgsz=CONFIG["imgsz"],
                        verbose=False,
                    )

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
                        elif self.date_id is not None and cls == self.date_id and conf >= self.current_checkpoint.get("conf_date", CONFIG.get("conf_date", 0.30)):
                            date_dets.append([int(x1), int(y1), int(x2), int(y2), conf])

                # ── Secondary date model inference (collect parallel result) ──
                secondary_date_dets = []
                if sec_future is not None:
                    try:
                        sec_results = sec_future.result(timeout=2.0)[0]
                        if sec_results.boxes is not None:
                            for b in sec_results.boxes:
                                cls = int(b.cls)
                                conf_val = float(b.conf)
                                if cls == self._secondary_date_id:
                                    sx1, sy1, sx2, sy2 = b.xyxy[0].cpu().numpy()
                                    secondary_date_dets.append([int(sx1), int(sy1), int(sx2), int(sy2), conf_val])
                    except Exception as sec_err:
                        print(f"[DETECTOR] Secondary date model error: {sec_err}")

                # Merge and deduplicate date detections
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
                require_date_for_ok = bool((self.current_checkpoint or {}).get("require_date_for_ok", False))

                for t in tracks:
                    x1, y1, x2, y2, tid = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])

                    if tid not in self.packages:
                        inherited_barcode = False
                        inherited_date = False
                        if not self._use_secondary_date:
                            new_bbox = (x1, y1, x2, y2)
                            for etid, epkg in self.packages.items():
                                if epkg.get("prev_bbox") and self._compute_iou(new_bbox, epkg["prev_bbox"]) > 0.3:
                                    if epkg.get("barcode_detected") and not inherited_barcode:
                                        inherited_barcode = True
                                    if epkg.get("date_detected") and not inherited_date:
                                        inherited_date = True
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

                    if not pkg["barcode_detected"]:
                        for bx1, by1, bx2, by2, bc in barcode_dets:
                            det_box = (bx1, by1, bx2, by2)
                            if self._det_box_matches_package(det_box, bbox, "barcode"):
                                pkg["barcode_detected"] = True
                                break

                    if not pkg.get("date_detected"):
                        for dx1, dy1, dx2, dy2, dc in all_date_dets:
                            det_box = (dx1, dy1, dx2, dy2)
                            if self._det_box_matches_package(det_box, bbox, "date"):
                                pkg["date_detected"] = True
                                break

                    if pkg["decision_locked"]:
                        color = (255, 165, 0)
                        status = pkg["final_decision"]
                    else:
                        has_barcode = pkg["barcode_detected"]
                        has_date = pkg.get("date_detected", False)
                        if require_date_for_ok:
                            if has_barcode and has_date:
                                color = (0, 255, 0)
                                status = "OK"
                            elif has_barcode:
                                color = (0, 165, 255)
                                status = "NOK(NO_DATE)"
                            else:
                                color = (0, 0, 255)
                                status = "NOK"
                        elif has_barcode:
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

                        current_exit = self._exit_line_y
                        line_is_vert = self._exit_line_vertical
                        exit_pct = self._exit_line_pct

                        # Snapshot the flag BEFORE potentially setting it —
                        # a track must have been seen before the line on a
                        # PREVIOUS frame to be eligible for crossing.
                        was_pre_line = pkg["pre_line_seen"]

                        if not was_pre_line:
                            effective_pct = (100 - exit_pct) if self._exit_line_inverted else exit_pct
                            if line_is_vert:
                                near_check = (x1 < current_exit) if effective_pct > 50 else (x1 > current_exit)
                            else:
                                near_check = (y1 < current_exit) if effective_pct > 50 else (y1 > current_exit)
                            if near_check:
                                pkg["pre_line_seen"] = True

                        if self._exit_line_inverted:
                            crossed_check = (x1 <= current_exit) if line_is_vert else (y1 <= current_exit)
                        else:
                            crossed_check = (x2 >= current_exit) if line_is_vert else (y2 >= current_exit)
                        # Guard: was_pre_line (not pkg["pre_line_seen"]) ensures
                        # approach and crossing never happen on the same frame.
                        # frames_tracked >= 3 filters out ByteTrack noise tracks.
                        if (crossed_check and was_pre_line
                                and pkg["frames_tracked"] >= 3
                                and tid not in self.packets_crossed_line):
                            if pkg["decision_locked"]:
                                self.packets_crossed_line.add(tid)
                                continue
                            self.packets_crossed_line.add(tid)
                            self.total_packets += 1
                            self.packet_numbers[tid] = self.total_packets
                            has_bc = pkg["barcode_detected"]
                            has_dt = pkg.get("date_detected", False)
                            final = "OK" if (has_bc and (has_dt or not require_date_for_ok)) else "NOK"
                            pkg["decision_locked"] = True
                            pkg["final_decision"] = final
                            self.output_fifo.append(final)

                            # Save proof image for defective packets
                            if final == "NOK":
                                if not has_bc:
                                    defect_type = "nobarcode"
                                elif require_date_for_ok and not has_dt:
                                    defect_type = "nodate"
                                else:
                                    defect_type = "nobarcode"
                                self._save_proof_image_bg(
                                    self.total_packets - self._session_baseline_total,
                                    defect_type, frame, (x1, y1, x2, y2))

                            # DB/session accounting is fully isolated: when
                            # recording is OFF, detection path does no DB work.
                            if self._stats_active:
                                if not has_bc:
                                    self._nok_no_barcode += 1
                                elif require_date_for_ok and not has_dt:
                                    self._nok_no_date += 1

                                if (
                                    self._db_writer
                                    and self._db_session_id
                                    and self.total_packets % SNAPSHOT_EVERY_N_PACKETS == 0
                                ):
                                    try:
                                        self._db_writer.write_queue.put_nowait({
                                            "type": "session_update",
                                            "session_id": self._db_session_id,
                                            "total": self.total_packets,
                                            "ok_count": self.output_fifo.count("OK"),
                                            "nok_no_barcode": self._nok_no_barcode,
                                            "nok_no_date": self._nok_no_date,
                                            "nok_anomaly": self._nok_anomaly,
                                        })
                                    except Exception:
                                        pass

                                # Record defective packet with timestamp for ejection
                                if final == "NOK":
                                    defect_type_db = "nobarcode"
                                    if not has_bc:
                                        defect_type_db = "nobarcode"
                                    elif require_date_for_ok and not has_dt:
                                        defect_type_db = "nodate"
                                    try:
                                        from datetime import datetime
                                        self._db_writer.write_queue.put_nowait({
                                            "type": "crossing",
                                            "session_id": self._db_session_id,
                                            "packet_num": self.total_packets - self._session_baseline_total,
                                            "defect_type": defect_type_db,
                                            "crossed_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
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

                # Barcode overlay
                barcode_vis = [(int(bx1), int(by1), int(bx2), int(by2), bc)
                               for bx1, by1, bx2, by2, bc in barcode_dets]

                # ── Store overlay for video feed ──
                with self._overlay_lock:
                    self._overlay = {
                        'track_boxes': track_boxes,
                        'barcode_boxes': barcode_vis,
                        'date_boxes': all_date_dets,
                        'exit_line_y': self._exit_line_y,
                        'total_packets': self.total_packets,
                        'fifo_str': fifo_str,
                        'det_fps': det_fps,
                        'det_ms': det_ms,
                        'frame_idx': frame_idx,
                    }

                # ── Update API stats ──
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
                with self._perf_lock:
                    self._perf["detector_loop_ms"] = round(det_ms, 2)

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
                loop_t0 = time.time()
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
                    ov_dates     = list(self._overlay.get('date_boxes', []))
                    ov_frame_idx = self._overlay.get('frame_idx', 0)
                    ov_total     = self._overlay.get('total_packets', 0)
                    ov_fifo      = self._overlay.get('fifo_str', '(empty)')
                    ov_det_fps   = self._overlay.get('det_fps', 0)
                    ov_det_ms    = self._overlay.get('det_ms', 0)
                    ov_ad_zones  = self._overlay.get('ad_zone_lines', None)

                # If available, draw overlay on the exact raw frame used by
                # detector to avoid apparent bbox shift on moving conveyor.
                if ov_frame_idx > 0:
                    matched = None
                    with self._raw_history_lock:
                        for idx, f in reversed(self._raw_history):
                            if idx == ov_frame_idx:
                                matched = f
                                break
                            if idx < ov_frame_idx:
                                break
                    if matched is not None:
                        frame = matched.copy()
                        h, w = frame.shape[:2]
                        with self._perf_lock:
                            self._perf["compositor_sync_hits"] += 1
                    else:
                        with self._perf_lock:
                            self._perf["compositor_sync_misses"] += 1

                # ── Exit line: use dedicated attribute (set once by detector, updated live) ──
                # Fall back to frame-based estimate only until detector computes it
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

                # ── Draw date boxes ──
                for (dx1, dy1, dx2, dy2, dc) in ov_dates:
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 0), 2)
                    cv2.putText(frame, f"date {dc:.2f}",
                                (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # ── Draw exit line (skip for anomaly mode — uses internal zone lines) ──
                if ov_ely > 0 and self._exit_line_enabled and self.mode != "anomaly":
                    if ov_line_vert:
                        cv2.line(frame, (ov_ely, 0), (ov_ely, h), (255, 0, 0), 3)
                        cv2.putText(frame, "EXIT LINE", (ov_ely + 5, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    else:
                        cv2.line(frame, (0, ov_ely), (w, ov_ely), (255, 0, 0), 3)
                        cv2.putText(frame, "EXIT LINE", (w - 200, ov_ely - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # ── Draw anomaly detection ENTRY / EXIT lines (vertical) ──
                if ov_ad_zones is not None:
                    exit_px, entry_px = ov_ad_zones
                    # Entry line (right side — where packets enter the scan zone)
                    cv2.line(frame, (entry_px, 0), (entry_px, h), (0, 200, 255), 2)
                    cv2.putText(frame, "ENTRY", (entry_px + 5, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    # Exit line (left side — decision is final here)
                    cv2.line(frame, (exit_px, 0), (exit_px, h), (255, 0, 0), 2)
                    cv2.putText(frame, "EXIT", (exit_px + 5, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

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

                with self._perf_lock:
                    self._perf["compositor_loop_ms"] = round((time.time() - loop_t0) * 1000, 2)

                # If video ended, keep last frame encoded and exit
                if self._video_ended and not self.is_running:
                    break

        except Exception as e:
            print(f"[COMPOSITOR] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[COMPOSITOR] Stopped")
