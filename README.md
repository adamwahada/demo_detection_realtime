# Real-Time Package Tracking & Barcode Detection

A parallel-architecture web server that performs real-time **package tracking** and **barcode detection**  and **date detection** using YOLO + ByteTrack. The video stream stays smooth regardless of YOLO inference speed.

## Architecture

```
Thread 1 — Reader      : reads video at native FPS (to ensure it's always smooth)
Thread 2 — Detector    : runs YOLO + ByteTrack in parallel (updates stats & overlays)
Thread 3 — Compositor  : composites raw frame + detection overlay, encodes JPEG
Web Feed               : serves pre-encoded JPEG via MJPEG (zero computation)
```

### Project Structure

```
web_server_backend_v2.py   ← Entry point (Flask routes + model init)
tracking_config.py         ← All tunable parameters (thresholds, FPS, image size…)
tracking_state.py          ← TrackingState class (reader, detector, compositor threads)
helpers.py                 ← Bbox metrics & fallen-package detection
templates/index.html       ← Web UI (dashboard + live video stream)
trackers/bytetrack/        ← ByteTrack multi-object tracker
  ├── bytetrack.py
  ├── kalman_filter.py
  └── matching.py
requirements_web.txt       ← Python dependencies
bestexp2.pt                ← YOLO model weights (not in repo — see Setup)
```

## Features

- **Live camera** (USB / V4L2), **RTSP streams**, or **video files** (.mp4, .avi, .mov…)
- Package classification:
  - **OK** — barcode detected on the package
  - **NOK** — no barcode detected when crossing the exit line
  - **DEFECTIVE** — package fell (sudden bbox area/aspect change)
- Real-time FIFO queue of decisions
- Web dashboard with live stats (FPS, inference time, packet counts)
- REST API for start/stop/stats/config

## Prerequisites

- **Python 3.8+**
- **NVIDIA GPU** with CUDA support
- **CUDA 11.7** + compatible drivers

## Setup

1. **Clone the repo:**
   ```bash
   git clone <repo-url>
   cd tracking_live
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements_web.txt
   ```
   > For PyTorch with CUDA 11.7, the packages `torch` and `torchvision` in `requirements_web.txt` point to the cu117 builds. If you use a different CUDA version, adjust accordingly.

4. **Add the model weights:**
   
   Place `bestexp2.pt` in the project root this chekpoint is for pacekts tracking and barcode detetcion the `yolo26-BB(date).pt` is the one for dates detecion and `yolo26m_BB_barcode_date.pt`is the one featuring dates and barcode and packet tracking 
## Running

```bash
python web_server_backend_v2.py
```

Open in your browser:
```
http://localhost:5000
```

Enter a video source in the UI and click **Start**:
- USB camera: `/dev/video0`
- Video file: `/path/to/video.mp4`

## Configuration

All parameters are in **`tracking_config.py`**:

| Parameter | Default | Description |
|---|---|---|
| `conf_paquet` | 0.5 | Minimum confidence for package detection |
| `conf_barcode` | 0.5 | Minimum confidence for barcode detection |
| `imgsz` | 416 | YOLO input image size |
| `track_thresh` | 0.5 | ByteTrack tracking threshold |
| `track_buffer` | 60 | Frames to keep lost tracks |
| `match_thresh` | 0.8 | ByteTrack matching threshold |
| `exit_line_ratio` | 0.15 | Exit line position (ratio from bottom) |
| `DETECTOR_FRAME_SKIP` | 3 | Send 1 frame out of N to the detector |
| `JPEG_QUALITY` | 80 | MJPEG stream quality |

You can also update config at runtime via the API:

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web dashboard |
| GET | `/video_feed` | MJPEG video stream |
| POST | `/api/start` | Start processing (`{"source": "/dev/video0"}`) |
| POST | `/api/stop` | Stop processing |
| GET | `/api/stats` | Current stats (JSON) |
| GET/POST | `/api/config` | Get or update config |
| GET | `/api/fifo` | Full FIFO queue |

## Troubleshooting

- **"Failed to fetch" in browser** — The server crashed. Check terminal for errors. Common cause: segfault from GPU memory (exit code 139).
- **Low YOLO FPS** — Increase `DETECTOR_FRAME_SKIP` or reduce `imgsz` in `tracking_config.py`.
- **Camera not opening** — Check `/dev/video0` exists (`ls /dev/video*`). Try a different index.
