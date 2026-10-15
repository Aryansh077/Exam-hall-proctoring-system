# 🎓 Real-Time Computer Vision Smart Exam Hall Proctoring System

An enterprise-grade, modular, real-time automated AI exam proctoring system built with **Python**, **OpenCV**, **MediaPipe Face Mesh**, and **Ultralytics YOLOv8**.

The system requires **zero custom dataset training** by strategically leveraging high-performance pre-trained foundation models.

---

## 🌟 Architecture & Core Features

```
+-----------------------------------------------------------------------------------------+
|                                Dynamic Video Ingestion                                  |
|         (Built-in Webcam / USB Camera / IP CCTV RTSP/HTTP Stream / Video File)          |
+-----------------------------------------------------------------------------------------+
                                             |
                      +----------------------+----------------------+
                      |                                             |
                      v                                             v
       +-------------------------------+             +-------------------------------+
       |        Ultralytics YOLOv8     |             |      MediaPipe Face Mesh      |
       |     (COCO Pretrained Weights) |             |    (468 3D CPU Landmarks)     |
       +-------------------------------+             +-------------------------------+
       | - Cell Phone (COCO ID 67)     |             | - Geometric Nose/Cheek Engine |
       | - Book / Notes (COCO ID 73)   |             | - Gaze Deviation (Left/Right) |
       | - Extra Persons (COCO ID 0)   |             | - Missing / Multiple Faces    |
       +-------------------------------+             +-------------------------------+
                      |                                             |
                      +----------------------+----------------------+
                                             |
                                             v
                      +---------------------------------------------+
                      |    False-Positive Temporal Streak Buffer    |
                      |     (15 Consecutive Frames Confirmation)    |
                      +---------------------------------------------+
                                             |
                      +----------------------+----------------------+
                      |                                             |
                      v                                             v
       +-------------------------------+             +-------------------------------+
       |    Timestamped Anomaly Logger |             |      On-Screen Head-Up HUD    |
       |  - violations/ Snapshots      |             |  - Dynamic Status Banner      |
       |  - proctoring_audit_log.csv   |             |  - Color-Coded Bounding Boxes |
       +-------------------------------+             |  - Live Telemetry Dashboard   |
                                                     +-------------------------------+
```

### 1. Dynamic Camera Ingestion (`camera.py`)
- Switch flexibly between:
  - Default laptop/built-in webcam: `0`
  - External USB cameras: `1`, `2`, ...
  - IP CCTV / Security camera streams: `rtsp://admin:password@192.168.1.100:554/h264Preview_01_main` or `http://...`
  - Recorded MP4/AVI exam footage for forensic post-processing.
- Equipped with **automatic reconnection backoff** for dropped RTSP streams and auto-looping for recorded files.

### 2. Zero-Shot Contraband Detection (`detector_yolo.py`)
- Pre-trained on standard 80-class COCO weights (`yolov8n.pt`):
  - **Cell Phone** (`COCO ID 67`): Tagged with high-visibility Red bounding boxes.
  - **Books / Study Notes** (`COCO ID 73`): Tagged with Orange bounding boxes.
  - **Extra Person / Impersonator / Helper** (`COCO ID 0`): Detects presence of second candidate or unauthorized bystander.

### 3. Geometric Head Pose Engine (`pose_estimator.py`)
- Utilizes MediaPipe's lightweight CPU 468 3D landmark mesh:
  - **Nose Tip**: Landmark index `1`
  - **Bilateral Cheek Markers**: Left cheek `234` and Right cheek `454`
- Computes normalized Euclidean distances:
  $$\Delta = \frac{d_{right} - d_{left}}{d_{right} + d_{left}}$$
  - **Forward / Clear**: $\Delta \approx 0$ ($|\Delta| \le 0.22$)
  - **Looking Left**: $\Delta > +0.22$
  - **Looking Right**: $\Delta < -0.22$
- Candidate Presence Tracking:
  - `0 Faces`: Triggers **`CANDIDATE_MISSING`**
  - `>1 Faces`: Triggers **`MULTIPLE_FACES`**

### 4. False-Positive Temporal Buffer (`temporal_buffer.py`)
- Natural blinks, quick eye twitches, or micro-adjustments do **not** trigger false alarms.
- Requires **15 consecutive frames** of continuous anomaly detection before confirmation.
- Includes configurable **cooldown hysteresis** to eliminate repetitive alert logging.

### 5. Academic Integrity Anomaly Logger (`logger.py`)
- Timestamped photographic evidence automatically stored in `violations/` (e.g. `20261014_091240_CELL_PHONE_DETECTED.jpg`).
- Append-only audit record saved to `proctoring_audit_log.csv` with schema:
  `Timestamp`, `Violation_Type`, `Confidence_Score`, `Streak_Frames`, `Snapshot_Filename`.

### 6. Modern Head-Up Display (HUD) (`hud.py`)
- **Top Dynamic Status Bar**:
  - 🟢 **Green**: Candidate verified clear (`STATUS: INTEGRITY VERIFIED (CLEAR)`).
  - 🟡 **Amber**: Streak accumulating (`WARNING: SUSPICIOUS ACTIVITY STREAK (n/15)`).
  - 🔴 **Red**: Active violation confirmed (`ALERT: VIOLATION CONFIRMED -> <TYPE>`).
- **Telemetry Dashboard**:
  - Real-time FPS & frame count.
  - Live head orientation and normalized gaze deviation index.
  - Active candidate count & detected threats.
  - Facial geometric vectors and bilateral cheek anchors.

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11 installed.

### 2. Virtual Environment Setup
A dedicated virtual environment `venv` is configured in the project root:

**Activate the Virtual Environment (Windows PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

*(Or on Command Prompt: `.\venv\Scripts\activate.bat`)*

### 3. Run Real-Time Proctoring Monitor
Default to built-in webcam (`0`):
```powershell
python main.py
```
*(Or run directly with the venv executable: `.\venv\Scripts\python.exe main.py`)*

### 4. Run With Custom Sources & Configurations

#### External USB Camera:
```bash
python main.py --source 1
```

#### IP CCTV RTSP Network Camera:
```bash
python main.py --source "rtsp://username:password@192.168.1.50:554/stream1"
```

#### Pre-recorded Video File:
```bash
python main.py --source "exam_session_room_2.mp4"
```

#### Adjust Sensitivity & Streak Thresholds:
```bash
python main.py --streak-threshold 20 --phone-conf 0.45 --gaze-threshold 0.25
```

---

## 🧪 Simulation & Automated Verification

If running in a test or headless environment without an active webcam, run the built-in simulation engine:

```bash
# Headless simulation
python demo_simulation.py

# With interactive window display
python demo_simulation.py --gui
```

This verifies all 4 dynamic scenarios (Clear, Gaze Deviation, Cell Phone, Candidate Missing), logs snapshots to `violations/`, and updates `proctoring_audit_log.csv`.

---

## ⌨️ Interactive Keyboard Controls

| Key | Action |
|:---:|:---|
| **`q`** / **`ESC`** | Gracefully releases camera, frees neural network memory, and exits |
| **`r`** | Resets active streak accumulator counters |

---

## 📁 Project Structure

```
d:/Projects/Computer Vision Minin project/
├── proctoring_system/
│   ├── __init__.py           # Package initializer
│   ├── config.py             # Global dataclass configuration and hyperparams
│   ├── camera.py             # Resilient camera ingestion (webcam, RTSP, video)
│   ├── detector_yolo.py      # Pretrained YOLOv8 contraband tracker
│   ├── pose_estimator.py     # MediaPipe Face Mesh geometric head pose engine
│   ├── temporal_buffer.py    # False-positive streak counter & anti-flicker
│   ├── logger.py             # CSV audit trail & snapshot exporter
│   ├── hud.py                # Futuristic HUD renderer & visual overlays
│   └── proctor.py            # ProctoringEngine coordinating pipeline
├── main.py                   # Main CLI entrypoint
├── demo_simulation.py        # Complete automated pipeline simulation
├── requirements.txt          # Python dependency specifications
├── proctoring_audit_log.csv  # Audit trail log file
└── violations/               # Photographic violation evidence archive
```
